import json
import os
import pprint
from collections import OrderedDict
from typing import Any, TypedDict

import torch
from safetensors.torch import load_file, save_file

from tilert import logger
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.models.deepseek_v3_2.model_args import ModelArgs as ModelArgsDsav32
from tilert.models.deepseek_v3_2.modules.mla import Mla
from tilert.models.deepseek_v3_2.ops.down_allreduce import DownAllReduce
from tilert.models.deepseek_v3_2.ops.eh_proj_allreduce import EHProjAllReduce
from tilert.models.deepseek_v3_2.ops.expert_down_allreduce import ExpertDownAllReduce
from tilert.models.deepseek_v3_2.ops.expert_sel_up_gate_silu import ExpertSelectUpGateSiLU
from tilert.models.deepseek_v3_2.ops.rmsnorm_head_proj import RMSNormHeadProj
from tilert.models.deepseek_v3_2.ops.rmsnorm_up_gate_silu import RMSNormUpGateSiLU
from tilert.models.glm_5.model_args import ModelArgsGLM5

__all__ = [
    "WeightConverter",
]


class ShardInfo(TypedDict):
    """Type definition for shard information."""

    filename: str
    tensors: list[str]


class WeightConverter:
    """Weight converter for DeepSeek V3.2 model."""

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        model_dir: str,
        save_dir: str,
        test_mode: bool = False,
    ) -> None:
        self.model_args = model_args
        self.num_devices = num_devices
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.test_mode = test_mode

        self.num_dense_layers = model_args.n_dense_layers
        self.num_moe_layers = model_args.n_layers - self.num_dense_layers
        self.num_mtp_layers = 1
        self.total_layers = self.num_dense_layers + self.num_moe_layers + self.num_mtp_layers
        if self.test_mode:
            self.target_layers = [0, self.model_args.n_dense_layers, self.model_args.n_layers]
        else:
            self.target_layers = list(range(self.total_layers))

        self.num_experts = model_args.n_routed_experts

        self.index_file = "model.safetensors.index.json"
        self.__check_dir()

        # specially treated the embedding, norm, and head weights
        # at the beginning and end of the model
        self.emb_name = "model.embed_tokens.weight"
        self.norm_name = "model.norm.weight"
        self.head_name = "lm_head.weight"
        self.special_treated_params: dict[str, str] = {}

        self.files_by_layers: dict[str, set[str]] = self.__group_by_layers()
        self.default_device = "cpu"

        self.converted_weights_dict: dict[str, OrderedDict[str, torch.Tensor]] = {}
        for i in range(self.num_devices):
            self.converted_weights_dict[f"dev_{i}"] = OrderedDict()

    def __get_layer_num(self, param: str) -> int:
        """Get layer number from parameter name."""
        if "layers" not in param:
            return -1
        try:
            return int(param.split(".")[2])
        except ValueError:
            raise ValueError(f"Invalid file name: {param}")

    def __group_by_layers(self) -> dict[str, set[str]]:
        """Load the index file."""
        with open(os.path.join(self.model_dir, self.index_file)) as f:
            weight_map = json.load(f)["weight_map"]

        files_by_layers: dict[str, set[str]] = {}
        for param, file_name in weight_map.items():
            layer_num = self.__get_layer_num(param)
            if layer_num == -1:
                logger.info(f"skip parameter {param} in {file_name}.")
                self.special_treated_params[param] = file_name
                continue

            key = f"layer_{layer_num}"
            if key in files_by_layers:
                files_by_layers[key].add(file_name)
            else:
                files_by_layers[key] = {file_name}

        return files_by_layers

    def __check_dir(self) -> None:
        if not os.path.exists(self.model_dir):
            raise ValueError(f"Model directory {self.model_dir} does not exist")

        if not os.path.exists(os.path.join(self.model_dir, self.index_file)):
            raise ValueError(f"Index file {self.index_file} not found in {self.model_dir}")

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def get_tensor_size_bytes(self, tensor: torch.Tensor) -> int:
        """Calculate the size of a tensor in bytes."""
        return int(tensor.numel() * tensor.element_size())

    def parse_size(self, size_str: str) -> int:
        """Parse size string like '1GB', '100MB' to bytes."""
        size_str = size_str.upper().strip()
        if size_str.endswith("GB"):
            return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
        if size_str.endswith("MB"):
            return int(float(size_str[:-2]) * 1024 * 1024)
        if size_str.endswith("KB"):
            return int(float(size_str[:-2]) * 1024)

        return int(size_str)

    def save_file_sharded(
        self,
        weights_dict: dict[str, torch.Tensor],
        base_filename: str,
        max_shard_size: str = "4GB",
        save_dir: str = "",
    ) -> list[ShardInfo]:
        """Save weights dictionary to multiple safetensors files.

        Each shard not exceeding max_shard_size.

        Args:
            weights_dict: Dictionary of tensor names to tensors
            base_filename: Base filename (e.g., "model.safetensors")
            max_shard_size: Maximum size per shard (e.g., "1GB", "100MB")
            save_dir: Directory to save the shards
        """
        if save_dir:
            base_filename = os.path.join(save_dir, base_filename)

        logger.info(f"Saving to safetensors format with max shard size {max_shard_size}...")

        max_size_bytes = self.parse_size(max_shard_size)

        tensor_nums = len(weights_dict)  # placeholder for number sharded files

        shards: list[ShardInfo] = []
        current_shard: dict[str, torch.Tensor] = {}
        current_size = 0
        shard_index = 1  # first shard is for embedding

        def get_shard_filename(shard_index: int) -> str:
            return f"{base_filename}-{shard_index:05d}-of-{tensor_nums:05d}.safetensors"

        # Save embedding tensor to separate file
        save_file(self.emb_weights_dict, get_shard_filename(shard_index))
        shards.append(
            {
                "filename": get_shard_filename(1),
                "tensors": list(self.emb_weights_dict.keys()),
            }
        )

        shard_index += 1
        for dev in weights_dict:
            logger.info(f"Processing weights for device {dev}")
            dev_tensors = weights_dict[dev]

            tensor_sizes = OrderedDict(
                {name: self.get_tensor_size_bytes(tensor) for name, tensor in dev_tensors.items()}
            )

            # If adding this tensor would exceed max size, start a new shard
            for tensor_name, tensor_size in tensor_sizes.items():
                if current_size + tensor_size > max_size_bytes and current_shard:
                    # Save current shard
                    shard_filename = get_shard_filename(shard_index)
                    logger.info(f"Saving shard {shard_index} to {shard_filename}")
                    save_file(current_shard, shard_filename)

                    shards.append(
                        {"filename": shard_filename, "tensors": list(current_shard.keys())}
                    )
                    current_shard = {}
                    current_size = 0
                    shard_index += 1

                # Add tensor to current shard
                current_shard[tensor_name] = dev_tensors[tensor_name]
                current_size += tensor_size

            # Save the last shard for the current device
            if current_shard:
                shard_filename = get_shard_filename(shard_index)
                logger.info(f"Saving shard {shard_index} to {shard_filename}")
                save_file(current_shard, shard_filename)
                shards.append({"filename": shard_filename, "tensors": list(current_shard.keys())})
                current_shard = {}
                current_size = 0
                shard_index += 1

        # Update shard filenames with correct total count
        total_shards = len(shards)
        for i, shard in enumerate(shards, 1):
            old_filename = shard["filename"]
            new_filename = f"{base_filename}-{i:05d}-of-{total_shards:05d}.safetensors"
            if old_filename != new_filename:
                os.rename(old_filename, new_filename)
                shard["filename"] = new_filename

        total_size = sum(self.get_tensor_size_bytes(t) for t in self.emb_weights_dict.values())
        for dev in weights_dict:
            dev_tensors = weights_dict[dev]
            tensor_sizes = OrderedDict(
                {name: self.get_tensor_size_bytes(tensor) for name, tensor in dev_tensors.items()}
            )
            total_size += sum(tensor_sizes.values())

        index: dict[str, Any] = {
            "metadata": {"total_size": total_size},
            "weight_map": {},
        }

        weight_map: dict[str, str] = index["weight_map"]  # type: ignore[assignment]
        for shard in shards:
            for tensor_name in shard["tensors"]:
                weight_map[tensor_name] = os.path.basename(shard["filename"])

        index_filename = f"{base_filename}.index.json"
        with open(index_filename, "w") as f:
            json.dump(index, f, indent=2)

        logger.info(f"Saved {total_shards} shard(s) with max size {max_shard_size}")
        logger.info(f"Index file: {index_filename}")
        return shards

    def transform_mla(
        self,
        weights_hf: dict[str, torch.Tensor],
        layer_id: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        mla_weights_map: dict[str, dict[str, torch.Tensor]] = {}
        for dev_id in range(self.num_devices):
            mla_weights_map.setdefault(f"dev_{dev_id}", {})
        mla = Mla(self.model_args, device_id=0, num_devices=self.num_devices)
        mla_raw_dict = {
            _k: weights_hf[f"model.layers.{layer_id}.{_k}"] for _k in mla.get_ref_weights_alias()
        }
        mla_sharded_dict = mla.device_sharding(mla_raw_dict)
        for dev_id in range(self.num_devices):
            for key, value in mla_sharded_dict.items():
                mla_weights_map[f"dev_{dev_id}"].update({key: value[dev_id].contiguous()})

        mla_weights = {}
        for dev_id in range(self.num_devices):
            mla_weights_dev = {}
            for key in mla_weights_map[f"dev_{dev_id}"].keys():
                mla_weights_dev.update({key: mla_weights_map[f"dev_{dev_id}"][key]})
            mla_weights.update({f"dev_{dev_id}": mla_weights_dev})

        return mla_weights

    def transform_moe(
        self,
        weights_hf: dict[str, torch.Tensor],
        layer_id: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        post_attn_norm_weight = f"model.layers.{layer_id}.post_attention_layernorm.weight"
        mlp_gate_weight = f"model.layers.{layer_id}.mlp.gate.weight"
        post_attn_norm_weight = weights_hf[post_attn_norm_weight].float()
        mlp_gate_weight = weights_hf[mlp_gate_weight]

        moe_weights: dict[str, dict[str, torch.Tensor]] = {}
        exp_sel_up_gate_silu = ExpertSelectUpGateSiLU(self.model_args, self.num_devices)
        ref_scope = f"model.layers.{layer_id}."
        exp_weights_map = {
            k: weights_hf[ref_scope + k] for k in exp_sel_up_gate_silu.ref_weights_alias()
        }
        exp_sharded = exp_sel_up_gate_silu.device_sharding(exp_weights_map)
        tilert_alias = exp_sel_up_gate_silu.tilert_weights_alias
        exp_bias = exp_sharded[tilert_alias.exp_bias]
        exp_gate_weights = exp_sharded[tilert_alias.exp_gate_weights]
        exp_gate_scales = exp_sharded[tilert_alias.exp_gate_scales]
        exp_up_weights = exp_sharded[tilert_alias.exp_up_weights]
        exp_up_scales = exp_sharded[tilert_alias.exp_up_scales]
        exp_down_allreduce = ExpertDownAllReduce(
            self.model_args, device_id=0, num_devices=self.num_devices
        )
        exp_down_weights, exp_down_scales = exp_down_allreduce.device_sharding(
            weights_hf, f"model.layers.{layer_id}.mlp"
        )
        for dev_id in range(self.num_devices):
            key = f"dev_{dev_id}"
            moe_weights.update(
                {
                    key: {
                        "unproj_o_gamma": post_attn_norm_weight,
                        "exp_proj_weights": mlp_gate_weight,
                        "exp_bias": exp_bias[dev_id],
                        "exp_gate_weights": exp_gate_weights[dev_id],
                        "exp_gate_scales": exp_gate_scales[dev_id],
                        "exp_up_weights": exp_up_weights[dev_id],
                        "exp_up_scales": exp_up_scales[dev_id],
                        "exp_down_weights": exp_down_weights[dev_id],
                        "exp_down_scales": exp_down_scales[dev_id],
                    }
                }
            )
        return moe_weights

    def transform_mlp(
        self,
        weights_hf: dict[str, torch.Tensor],
        layer_id: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Transform MLP weights."""
        print(RMSNormUpGateSiLU)
        rmsnorm_up_gate_silu = RMSNormUpGateSiLU(
            self.model_args, device_id=0, num_devices=self.num_devices
        )
        post_attn_norm_weight, gate_weights, gate_scales, up_weights, up_scales = (
            rmsnorm_up_gate_silu.device_sharding(weights_hf, f"model.layers.{layer_id}.mlp")
        )
        down_allreduce = DownAllReduce(self.model_args, device_id=0, num_devices=self.num_devices)
        down_weights, down_scales = down_allreduce.device_sharding(
            weights_hf, f"model.layers.{layer_id}.mlp"
        )

        weights_unproj_o_gamma: dict[str, dict[str, torch.Tensor]] = {}
        for dev_id in range(self.num_devices):
            weights_unproj_o_gamma[f"dev_{dev_id}"] = {
                "unproj_o_gamma": post_attn_norm_weight[dev_id]
            }

        weights_upgate: dict[str, dict[str, torch.Tensor]] = {}
        for dev_id in range(self.num_devices):
            weights_upgate.update(
                {
                    f"dev_{dev_id}": {
                        "gate_weights": gate_weights[dev_id],
                        "gate_scales": gate_scales[dev_id],
                        "up_weights": up_weights[dev_id],
                        "up_scales": up_scales[dev_id],
                    }
                }
            )

        weights_down: dict[str, dict[str, torch.Tensor]] = {}
        for dev_id in range(self.num_devices):
            weights_down.update(
                {
                    f"dev_{dev_id}": {
                        "down_weights": down_weights[dev_id],
                        "down_scales": down_scales[dev_id],
                    }
                }
            )

        mlp_weights: dict[str, dict[str, torch.Tensor]] = {}
        for dev_id in range(self.num_devices):
            mlp_weights_dev: dict[str, torch.Tensor] = {}
            mlp_weights_dev.update(weights_unproj_o_gamma[f"dev_{dev_id}"])
            mlp_weights_dev.update(weights_upgate[f"dev_{dev_id}"])
            mlp_weights_dev.update(weights_down[f"dev_{dev_id}"])
            mlp_weights[f"dev_{dev_id}"] = mlp_weights_dev
        return mlp_weights

    def transform_mtp(
        self,
        weights_hf: dict[str, torch.Tensor],
        layer_id: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Transform MTP weights.

        Transformations applied:
        - enorm.weight: Direct use (fp32)
        - hnorm.weight: Direct use (fp32)
        - eh_proj.weight: Split along dim 1, reshape [7168, 1792] -> [128, 7, 56, 256]
        """
        enorm_weight_key = f"model.layers.{layer_id}.enorm.weight"
        hnorm_weight_key = f"model.layers.{layer_id}.hnorm.weight"
        enorm_weight = weights_hf[enorm_weight_key]
        hnorm_weight = weights_hf[hnorm_weight_key]

        eh_proj_allreduce = EHProjAllReduce(self.model_args, self.num_devices)
        (eh_proj_weights,) = eh_proj_allreduce.device_sharding(
            weights_hf, f"model.layers.{layer_id}"
        )

        return {
            f"dev_{dev_id}": {
                "embedding_rmsnorm_gamma": enorm_weight,
                "hidden_rmsnorm_gamma": hnorm_weight,
                "eh_proj_weights": eh_proj_weights[dev_id],
            }
            for dev_id in range(self.num_devices)
        }

    def convert_a_layer(self, layer_idx: int) -> tuple[
        dict[str, dict[str, torch.Tensor]],
        dict[str, dict[str, torch.Tensor]],
        dict[str, dict[str, torch.Tensor]],
    ]:
        assert layer_idx < self.total_layers

        key = f"layer_{layer_idx}"
        files_to_load = self.files_by_layers[key]

        weights_dict = {}
        for file_name in files_to_load:
            logger.info(f"Loading weight from {file_name}")
            path = os.path.join(self.model_dir, file_name)
            weights = load_file(path, device=self.default_device)
            weights_dict.update(weights)

        mla_weights = self.transform_mla(weights_dict, layer_idx)

        if layer_idx < self.num_dense_layers:
            mlp_weights = self.transform_mlp(weights_dict, layer_idx)
        else:
            mlp_weights = self.transform_moe(weights_dict, layer_idx)

        mtp_weights: dict[str, dict[str, torch.Tensor]] = {
            f"dev_{dev_id}": {} for dev_id in range(self.num_devices)
        }
        if layer_idx >= self.num_dense_layers + self.num_moe_layers:
            mtp_weights = self.transform_mtp(weights_dict, layer_idx)

        return mla_weights, mlp_weights, mtp_weights

    def __process_head_weights(self) -> None:
        """Process head weights."""
        head_weight_file = self.special_treated_params[self.head_name]
        head_weight_file = os.path.join(self.model_dir, head_weight_file)
        head_weights = load_file(head_weight_file, device=self.default_device)[self.head_name]

        norm_weight_file = self.special_treated_params[self.norm_name]
        norm_weight_file = os.path.join(self.model_dir, norm_weight_file)
        norm_weights = load_file(norm_weight_file, device=self.default_device)[self.norm_name]

        weights_hf = {
            "model.norm.weight": norm_weights,
            "lm_head.weight": head_weights,
        }

        layer_idx = self.num_dense_layers + self.num_moe_layers
        rmsnorm_head_proj = RMSNormHeadProj(
            self.model_args, device_id=0, num_devices=self.num_devices
        )
        gamma, head_proj = rmsnorm_head_proj.device_sharding(weights_hf)

        for dev_id in range(self.num_devices):
            self.converted_weights_dict[f"dev_{dev_id}"][
                f"layer_{layer_idx}_lm_head.weight_dev_{dev_id}"
            ] = head_proj[dev_id]
            self.converted_weights_dict[f"dev_{dev_id}"][
                f"layer_{layer_idx}_model.norm.weight_dev_{dev_id}"
            ] = gamma[dev_id]

    def __process_embedding_weights(self) -> None:
        """Process embedding weights."""
        embedding_weight_file = self.special_treated_params[self.emb_name]
        embedding_weight_file = os.path.join(self.model_dir, embedding_weight_file)
        embedding_weights = load_file(embedding_weight_file, device=self.default_device)[
            self.emb_name
        ]
        self.emb_weights_dict = {"model.embed_tokens.weight": embedding_weights}

    def __post_process_weights(
        self,
        mla_weights: dict[str, dict[str, torch.Tensor]],
        mlp_weights: dict[str, dict[str, torch.Tensor]],
        mtp_weights: dict[str, dict[str, torch.Tensor]],
        layer_idx: int,
    ) -> None:
        """Post process weights."""
        for weights_group in [mla_weights, mlp_weights, mtp_weights]:
            for dev, params in weights_group.items():
                for param_name, tensor in params.items():
                    new_key = f"layer_{layer_idx}_{param_name}_{dev}"
                    self.converted_weights_dict[dev][new_key] = tensor

    def to_tilert_weights(self) -> None:
        torch.set_default_device(self.default_device)

        for i in range(self.total_layers):
            if i not in self.target_layers:
                logger.info(f"Skipping layer {i + 1} / {self.total_layers}")
                continue
            logger.info(f"Converting weight layer {i + 1} / {self.total_layers}")

            mla_weights, mlp_weights, mtp_weights = self.convert_a_layer(i)
            self.__post_process_weights(mla_weights, mlp_weights, mtp_weights, i)

        self.__process_head_weights()
        self.__process_embedding_weights()

        def _get_layer_num(file_name: str) -> tuple[int, int]:
            """Extract layer number from filename like 'layer_XX.xxx'."""
            if "/" in file_name:
                file_name = file_name.split("/")[-1]

            parts = file_name.split("_")
            try:
                layer_num = int(parts[1])
            except ValueError:
                raise ValueError(f"Could not find layer number in parameter name: {file_name}")
            try:
                device_id = int(parts[-1])
            except ValueError:
                raise ValueError(f"Could not find device id in parameter name: {file_name}")
            return (device_id, layer_num)

        def _sort_key(filename: str) -> tuple[int, int]:
            """Sort key function that returns (layer_num, device_id)."""
            try:
                return _get_layer_num(filename)
            except ValueError:
                return (999999, 999999)  # If layer number not found, put at the end

        tilert_weights = sorted(
            self.converted_weights_dict, key=lambda x: _sort_key(x), reverse=False
        )
        pprint.pprint(tilert_weights)  # noqa: T203

        self.save_file_sharded(
            self.converted_weights_dict,
            "model.safetensors",
            max_shard_size="5GB",
            save_dir=self.save_dir,
        )

    def append_mtp_weights_to_safetensors(
        self,
        existing_save_dir: str,
        max_shard_size: str = "5GB",
    ) -> None:
        """Append MTP layer weights to existing safetensors files.

        This method is used when layer 0-60 weights have already been converted,
        and we only need to add the MTP layer (layer 61) weights.

        Note: lm_head.weight and model.norm.weight are already included in the
        existing safetensors (converted with layer 0-60), so we only append:
        - MTP preprocess weights (enorm, hnorm, eh_proj)
        - MTP MLA weights
        - MTP MoE weights

        Args:
            existing_save_dir: Directory containing existing converted weights
            max_shard_size: Maximum shard size for new safetensors files
        """
        torch.set_default_device(self.default_device)

        # Load existing index.json
        existing_index_file = os.path.join(existing_save_dir, "model.safetensors.index.json")
        if not os.path.exists(existing_index_file):
            raise ValueError(f"Existing index file not found: {existing_index_file}")

        with open(existing_index_file) as f:
            existing_index = json.load(f)

        existing_weight_map: dict[str, str] = existing_index["weight_map"]
        existing_total_size: int = existing_index["metadata"]["total_size"]

        # Find the next shard number
        existing_shards = set(existing_weight_map.values())
        max_shard_num = 0
        for shard_name in existing_shards:
            # Parse shard number from filename like "model.safetensors-00001-of-00010.safetensors"
            parts = shard_name.replace(".safetensors", "").split("-")
            if len(parts) >= 2:
                try:
                    shard_num = int(parts[-2])
                    max_shard_num = max(max_shard_num, shard_num)
                except ValueError:
                    pass

        logger.info(
            f"Found {len(existing_shards)} existing shards, max shard number: {max_shard_num}"
        )

        # Convert MTP layer (layer 61) weights
        mtp_layer_idx = self.num_dense_layers + self.num_moe_layers  # 61
        logger.info(f"Converting MTP layer {mtp_layer_idx} weights...")

        mla_weights, mlp_weights, mtp_weights = self.convert_a_layer(mtp_layer_idx)

        # Collect MTP layer weights for all devices
        # Clone tensors to avoid shared memory issues when saving to safetensors
        mtp_layer_weights: dict[str, torch.Tensor] = {}
        for weights_group in [mla_weights, mlp_weights, mtp_weights]:
            for dev, params in weights_group.items():
                for param_name, tensor in params.items():
                    new_key = f"layer_{mtp_layer_idx}_{param_name}_{dev}"
                    mtp_layer_weights[new_key] = tensor.clone()

        logger.info(f"Collected {len(mtp_layer_weights)} MTP layer weight tensors")

        # Calculate size of new weights
        new_weights_size = sum(self.get_tensor_size_bytes(t) for t in mtp_layer_weights.values())

        # Save MTP weights to new shard file(s)
        # Use a separate naming scheme to avoid modifying existing shards
        max_size_bytes = self.parse_size(max_shard_size)
        new_shards: list[ShardInfo] = []
        current_shard: dict[str, torch.Tensor] = {}
        current_size = 0
        mtp_shard_index = 1  # Start from 1 for MTP shards

        for tensor_name, tensor in mtp_layer_weights.items():
            tensor_size = self.get_tensor_size_bytes(tensor)
            if current_size + tensor_size > max_size_bytes and current_shard:
                # Save current shard with MTP-specific naming
                shard_filename = f"model_mtp_layer61-{mtp_shard_index:05d}.safetensors"
                shard_path = os.path.join(existing_save_dir, shard_filename)
                logger.info(f"Saving MTP shard to {shard_filename}")
                save_file(current_shard, shard_path)
                new_shards.append(
                    {"filename": shard_filename, "tensors": list(current_shard.keys())}
                )
                current_shard = {}
                current_size = 0
                mtp_shard_index += 1

            current_shard[tensor_name] = tensor
            current_size += tensor_size

        # Save the last shard
        if current_shard:
            shard_filename = f"model_mtp_layer61-{mtp_shard_index:05d}.safetensors"
            shard_path = os.path.join(existing_save_dir, shard_filename)
            logger.info(f"Saving MTP shard to {shard_filename}")
            save_file(current_shard, shard_path)
            new_shards.append({"filename": shard_filename, "tensors": list(current_shard.keys())})

        # Update weight_map with new MTP weights (existing shards remain unchanged)
        for shard in new_shards:
            for tensor_name in shard["tensors"]:
                existing_weight_map[tensor_name] = shard["filename"]

        # Update index.json
        updated_index = {
            "metadata": {"total_size": existing_total_size + new_weights_size},
            "weight_map": existing_weight_map,
        }

        with open(existing_index_file, "w") as f:
            json.dump(updated_index, f, indent=2)

        logger.info(f"Added {len(new_shards)} new MTP shard(s)")
        logger.info(f"New total size: {existing_total_size + new_weights_size}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--test_mode", action="store_true", help="Test mode")
    parser.add_argument(
        "--append_mtp",
        action="store_true",
        help="Append MTP layer (layer 61) weights to existing safetensors. "
        "Use this when layer 0-60 weights have already been converted.",
    )
    args = parser.parse_args()

    model_type = args.model_type
    if model_type == "deepseek-v32":
        model_args = ModelArgsDsav32()
    elif model_type == "glm-5":
        model_args = ModelArgsGLM5()
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    converter = WeightConverter(model_args, 8, args.model_dir, args.save_dir, args.test_mode)
    if args.append_mtp:
        converter.append_mtp_weights_to_safetensors(args.save_dir)
    else:
        converter.to_tilert_weights()
