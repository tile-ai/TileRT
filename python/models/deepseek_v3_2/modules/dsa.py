from typing import Any

import torch

from tilert.models.base import SerializableTileRTModule
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.models.deepseek_v3_2.modules.mlp import MlpBlock
from tilert.models.deepseek_v3_2.modules.moe import MoeBlock
from tilert.models.deepseek_v3_2.ops import RMSNormHeadProj
from tilert.models.deepseek_v3_2.temp_var_indices import TEMP_VARS_SIZE, Idx


class Dsa(SerializableTileRTModule):
    """DSA module."""

    def __init__(self, model_args: ModelArgs, device_id: int, num_devices: int):
        super().__init__(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            remove_selected=True,
        )

        for layer_idx in range(model_args.n_layers):
            block_type = MlpBlock if layer_idx < model_args.n_dense_layers else MoeBlock
            block = block_type(model_args=model_args, device_id=device_id, num_devices=num_devices)
            self.register_op(block, prefix=f"layer_{layer_idx}_", suffix=f"_dev_{device_id}")

        self.register_op(
            RMSNormHeadProj(model_args=model_args, device_id=device_id, num_devices=num_devices),
            prefix=f"layer_{model_args.n_layers}_",
            suffix=f"_dev_{device_id}",
            retain_weights=True,
        )

        self.embed_tokens_weight = None
        self.freqs_cis = None

    def init_tilert_weights(self, state_dicts: dict[str, torch.Tensor]) -> None:
        super().init_tilert_weights(state_dicts)
        self.embed_tokens_weight = state_dicts["model.embed_tokens.weight"]
        self.freqs_cis = state_dicts["freqs_cis"]

    def get_weights_list(self) -> list[torch.Tensor]:
        return [*super().get_weights_list(), self.embed_tokens_weight, self.freqs_cis]

    def get_temp_vars(
        self, batch_size: int, seq_len: int, extra_args: dict[str, Any] | None = None
    ) -> list[torch.Tensor]:
        bf16_desc = {"dtype": torch.bfloat16, "device": f"cuda:{self.device_id}"}
        fp32_desc = {"dtype": torch.float32, "device": f"cuda:{self.device_id}"}
        int32_desc = {"dtype": torch.int32, "device": f"cuda:{self.device_id}"}
        int64_desc = {"dtype": torch.int64, "device": f"cuda:{self.device_id}"}
        fp8_desc = {"dtype": torch.float8_e4m3fn, "device": f"cuda:{self.device_id}"}

        assert extra_args is not None
        temperature = extra_args["temperature"]
        top_p = extra_args["top_p"]
        top_k = extra_args["top_k"]
        use_topp = extra_args["use_topp"]

        dim = self.model_args.dim
        batch_seq = (batch_size, seq_len)
        q_lora_rank = self.model_args.q_lora_rank
        kv_lora_rank = self.model_args.kv_lora_rank
        qk_nope_head_dim = self.model_args.qk_nope_head_dim
        n_local_heads = self.model_args.n_heads // self.num_devices
        qk_rope_head_dim = self.model_args.qk_rope_head_dim
        index_head_dim = self.model_args.index_head_dim
        v_head_dim = self.model_args.v_head_dim
        n_index_heads = self.model_args.index_n_heads
        max_seq_len = self.model_args.max_seq_len
        index_topk = self.model_args.index_topk
        n_routed_experts = self.model_args.n_routed_experts
        n_activated_experts = self.model_args.n_activated_experts
        n_total_experts = self.model_args.n_activated_experts + self.model_args.n_shared_experts
        moe_inter_dim = self.model_args.moe_inter_dim // self.num_devices
        vocab_size = self.model_args.vocab_size // self.num_devices

        temp_vars: list[torch.Tensor | None] = [None] * TEMP_VARS_SIZE

        temp_vars[Idx.Q] = torch.zeros(*batch_seq, q_lora_rank, **bf16_desc)
        temp_vars[Idx.KV] = torch.zeros(*batch_seq, kv_lora_rank, **bf16_desc)
        temp_vars[Idx.KI] = torch.zeros(*batch_seq, index_head_dim, **bf16_desc)
        temp_vars[Idx.Q_NOPE_DOWN] = torch.zeros(
            *batch_seq, n_local_heads, qk_nope_head_dim, **bf16_desc
        )
        temp_vars[Idx.Q_PE] = torch.zeros(*batch_seq, n_local_heads, qk_rope_head_dim, **bf16_desc)
        temp_vars[Idx.IQ] = torch.zeros(*batch_seq, n_index_heads, index_head_dim, **bf16_desc)
        temp_vars[Idx.IQ_RT] = torch.zeros(*batch_seq, n_index_heads, index_head_dim, **bf16_desc)
        temp_vars[Idx.IDX_SCORES] = torch.zeros(*batch_seq, n_index_heads, **bf16_desc)
        temp_vars[Idx.IDX_LOGITS] = torch.zeros(
            *batch_seq, max_seq_len + self.model_args.kv_cache_pad, **fp32_desc
        )
        temp_vars[Idx.IDX_SELECTS] = torch.zeros(*batch_seq, index_topk, **int32_desc)
        temp_vars[Idx.Q_NOPE] = torch.zeros(*batch_seq, n_local_heads, kv_lora_rank, **bf16_desc)
        temp_vars[Idx.O] = torch.zeros(*batch_seq, n_local_heads, kv_lora_rank, **bf16_desc)
        temp_vars[Idx.O_ACC] = torch.zeros(*batch_seq, n_local_heads, 32, kv_lora_rank, **fp32_desc)
        temp_vars[Idx.O_LSE] = torch.empty(*batch_seq, n_local_heads, **fp32_desc)
        temp_vars[Idx.O_LSE_ACC] = torch.empty(*batch_seq, n_local_heads, 32, **fp32_desc)
        temp_vars[Idx.PROJ_O] = torch.zeros(*batch_seq, n_local_heads, v_head_dim, **bf16_desc)
        temp_vars[Idx.UNPROJ_O] = torch.zeros(*batch_seq, dim, **bf16_desc)
        temp_vars[Idx.SCORES] = torch.zeros(*batch_seq, n_routed_experts, **fp32_desc)
        temp_vars[Idx.X_MLP_IN] = torch.zeros(*batch_seq, dim, **bf16_desc)
        exp_up_gate = torch.zeros(*batch_seq, n_total_experts, moe_inter_dim, **bf16_desc)
        temp_vars[Idx.UP_GATE] = exp_up_gate
        temp_vars[Idx.SEL_PROBS] = torch.zeros(*batch_seq, n_activated_experts, **fp32_desc)
        temp_vars[Idx.SEL_INDICES] = torch.zeros(*batch_seq, n_activated_experts, **int32_desc)
        temp_vars[Idx.EXP_OUT] = torch.zeros(*batch_seq, dim, **bf16_desc)
        temp_vars[Idx.X_RMSNORM] = torch.zeros(*batch_seq, dim, **bf16_desc)
        temp_vars[Idx.LOGITS_OUT] = torch.zeros(*batch_seq, vocab_size, **fp32_desc)
        temp_vars[Idx.TOKEN_OUT] = torch.zeros(*batch_seq, 1, **int32_desc)

        temp_vars[Idx.EMBEDDING_RMSNORM] = torch.zeros(*batch_seq, dim, **bf16_desc)
        temp_vars[Idx.HIDDEN_RMSNORM] = torch.zeros(*batch_seq, dim, **bf16_desc)
        temp_vars[Idx.EH_PROJ] = torch.zeros(*batch_seq, dim, **bf16_desc)
        temp_vars[Idx.X_TENSOR] = torch.zeros(*batch_seq, dim, **bf16_desc)
        temp_vars[Idx.ROPE_FREQS] = torch.zeros(*batch_seq, qk_rope_head_dim, **fp32_desc)
        temp_vars[Idx.CUR_POS] = torch.zeros(batch_size, **int32_desc)
        temp_vars[Idx.TOKEN_ID] = torch.zeros(*batch_seq, 1, **int32_desc)
        temp_vars[Idx.LAST_HIDDEN_STATES] = torch.zeros(*batch_seq, dim, **bf16_desc)

        temp_vars[Idx.DRAFT_TOKENS] = torch.zeros(*batch_seq, **int32_desc)
        temp_vars[Idx.PREDICTED_TOKENS] = torch.zeros(*batch_seq, 1, **int32_desc)
        temp_vars[Idx.PREDICTED_HIDDEN] = torch.zeros(*batch_seq, dim, **bf16_desc)
        temp_vars[Idx.ACCEPTED_TOKENS] = torch.zeros(batch_size, **int32_desc)
        temp_vars[Idx.NEXT_DRAFT_TOKENS] = torch.zeros(*batch_seq, **int32_desc)

        temp_vars[Idx.X_QUANT] = torch.zeros(*batch_seq, dim, **fp8_desc)
        temp_vars[Idx.X_SCALE] = torch.zeros(
            *batch_seq, dim // self.model_args.block_size, **fp32_desc
        )
        temp_vars[Idx.MOE_UP_GATE] = torch.zeros_like(exp_up_gate)

        # temp_vars[Idx.IDX_SEL_WS] = torch.zeros(*batch_seq, 4, index_topk * 2, **int32_desc)
        temp_vars[Idx.IDX_SEL_WS] = torch.zeros(*batch_seq, (200 * 1024 + 258), **int32_desc)

        temp_vars[Idx.MTP0_TOKEN_OUT] = torch.zeros(*batch_seq, 1, **int32_desc)
        temp_vars[Idx.MTP1_TOKEN_OUT] = torch.zeros(*batch_seq, 1, **int32_desc)
        temp_vars[Idx.MTP0_EXP_OUT] = torch.zeros(*batch_seq, dim, **bf16_desc)

        temp_vars[Idx.SAMPLING_SEED] = torch.zeros(*batch_seq, **int64_desc)
        temp_vars[Idx.SAMPLING_POSITIONS] = torch.zeros(*batch_seq, **int64_desc)
        temp_vars[Idx.SAMPLING_CONFIG] = torch.tensor(
            [temperature, top_p, top_k, use_topp], **fp32_desc
        )
        temp_vars[Idx.TOP_P_SCORES] = torch.zeros(*batch_seq, **fp32_desc)
        temp_vars[Idx.TOP_P_DEBUG] = torch.zeros(*batch_seq, vocab_size, **fp32_desc)

        for i, t in enumerate(temp_vars):
            if t is None:
                raise RuntimeError(f"temp_vars[{i}] ({Idx(i).name}) was not initialized")

        return temp_vars  # type: ignore[return-value]
