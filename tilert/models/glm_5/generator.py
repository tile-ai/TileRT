"""DSA show hands for GLM5."""

import os
import time

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from tilert import logger
from tilert.models.glm_5._dsa_v32.model_args import ModelArgs
from tilert.models.glm_5.modules.end2end import ShowHandsDSALayer
from tilert.models.glm_5.temp_var_indices import Idx
from tilert.tilert_init import tilert_init

__all__ = [
    "GLM5Generator",
]


class GLM5Generator:
    """Show hands generator for GLM5."""

    def __init__(
        self,
        model_args: ModelArgs,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        model_weights_dir: str = "",
        with_mtp: bool = False,
        top_p: float = 0.9,
        top_k: int = 256,
        use_topp: bool = False,
        enable_thinking: bool = False,
        sampling_seed: int = 42,
    ):
        """Initialize the ShowHandsGeneratorGlm5."""
        torch.set_num_threads(64)
        self.model_weights_dir = model_weights_dir

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.with_mtp = with_mtp
        self.enable_thinking = enable_thinking
        self.sampling_seed = sampling_seed

        self.config = model_args
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_weights_dir, trust_remote_code=True
            )  # nosec B615
        except (ValueError, KeyError):
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                self.model_weights_dir, trust_remote_code=True
            )  # nosec B615
        jinja_file_path = os.path.join(self.model_weights_dir, "chat_template.jinja")
        with open(jinja_file_path, encoding="utf-8") as f:
            chat_template = f.read()
        self.tokenizer.chat_template = chat_template
        self.eos_id = self.tokenizer.eos_token_id
        self.batch_size = 1
        self.mtp_seq_len = 4

        self.stop_tokens = ["<|user|>", "<|endoftext|>", "<|observation|>", "<|assistant|>"]
        self.stop_token_ids: set[int] = set()
        for token in self.stop_tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) == 1:
                self.stop_token_ids.add(token_ids[0])
            else:
                if (
                    hasattr(self.tokenizer, "added_tokens_encoder")
                    and token in self.tokenizer.added_tokens_encoder
                ):
                    self.stop_token_ids.add(self.tokenizer.added_tokens_encoder[token])
        if self.eos_id is not None:
            self.stop_token_ids.add(self.eos_id)
        logger.info(f"Stop token IDs: {self.stop_token_ids}")

        self.default_device = torch.device("cuda:0")

        self.decode_layer = ShowHandsDSALayer(
            model_args=self.config,
            model_path=self.model_weights_dir,
            with_mtp=with_mtp,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_topp=use_topp,
        )

    def init(self) -> None:
        """Initialize the ShowHandsGeneratorGlm5."""
        tilert_init()

    def cleanup(self) -> None:
        """Cleanup the ShowHandsGeneratorGlm5."""
        self.decode_layer.cleanup()

    def init_random_weights(self) -> None:
        """Random initialize the weights."""
        self.decode_layer.init_random_weights()

    def from_pretrained(self) -> None:
        """Load the model weights from the given path."""
        self.decode_layer.from_pretrained(self.model_weights_dir)

    def extract_ffn_cache(self) -> tuple[dict[int, list], dict[int, set[str]]]:
        """Extract MOE/MLP op objects and skip keys from current loaded weights."""
        from tilert.models.glm_5.modules.end2end import (
            _extract_ffn_ops,
            _get_moe_weight_keys,
        )

        cached_ffn_ops: dict[int, list] = {}
        skip_keys: dict[int, set[str]] = {}
        for device_id in range(self.decode_layer.num_devices):
            dsa = self.decode_layer._dsa_objects[device_id]
            if dsa is None:
                raise RuntimeError(f"Device {device_id} Dsa not available for cache extraction")
            cached_ffn_ops[device_id] = _extract_ffn_ops(dsa)
            skip_keys[device_id] = _get_moe_weight_keys(dsa)
        return cached_ffn_ops, skip_keys

    def from_pretrained_with_cache(
        self,
        cached_ffn_ops_per_device: dict[int, list],
        skip_keys_per_device: dict[int, set[str]],
    ) -> None:
        """Load weights reusing cached MOE/MLP ops."""
        self.decode_layer.from_pretrained_with_cache(
            self.model_weights_dir, cached_ffn_ops_per_device, skip_keys_per_device
        )

    def update_sampling_params(
        self,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 256,
        use_topp: bool = True,
    ) -> None:
        """Update sampling parameters for the next generation."""
        self.temperature = temperature
        self.decode_layer.update_sampling_config(
            temperature=temperature, top_p=top_p, top_k=top_k, use_topp=use_topp
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        print_log: bool = True,
        with_mtp: bool | None = None,
        prompt_tokens: list[int] | None = None,
    ) -> tuple[str, list[float], list[int], int]:
        """Main function to load the model and perform single sequence generation."""
        active_mtp = with_mtp if with_mtp is not None else self.with_mtp
        if active_mtp and not self.with_mtp:
            raise ValueError("Cannot use MTP mode: MTP weights were not loaded")
        self.decode_layer.set_sampling_seed(self.sampling_seed, with_mtp=active_mtp)
        if active_mtp:
            return self._generate_with_mtp(prompt, print_log, prompt_tokens=prompt_tokens)
        result, time_list, prompt_len = self._generate_without_mtp(
            prompt, print_log, prompt_tokens=prompt_tokens
        )
        return result, time_list, [], prompt_len

    def _generate_without_mtp(
        self,
        prompt: str,
        print_log: bool = True,
        prompt_tokens: list[int] | None = None,
    ) -> tuple[str, list[float], int]:
        """Standard generation without MTP (unified single-op decode)."""
        if prompt_tokens is None:
            messages = [{"role": "user", "content": prompt}]
            prompt_tokens = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                return_dict=False,
            )

        max_seq_len = self.config.max_seq_len
        prompt_len = len(prompt_tokens)
        total_len = min(max_seq_len, self.max_new_tokens + prompt_len)

        tokens = torch.full(
            (self.batch_size, total_len), -1, dtype=torch.long, device=self.default_device
        )
        tokens[0, :prompt_len] = torch.tensor(
            prompt_tokens, dtype=torch.long, device=self.default_device
        )

        ar_steps = max(1, min(1024, int(os.environ.get("GLM5_AR_N", "8"))))
        return self._decode_without_mtp_ar(tokens, prompt_len, total_len, ar_steps, print_log)

    def _decode_without_mtp_ar(
        self,
        tokens: torch.Tensor,
        prompt_len: int,
        total_len: int,
        ar_steps: int,
        print_log: bool,
    ) -> tuple[str, list[float], int]:
        """w/o-MTP decode (unified single-op, unified-style)."""
        time_list: list[float] = []

        self.decode_layer.set_prefill_valid_tokens(1, with_mtp=False)
        for prev_pos in range(prompt_len - 1):
            self.decode_layer.forward(tokens[0, prev_pos], with_mtp=False)
        self.decode_layer.set_prefill_valid_tokens(0, with_mtp=False)

        cur_pos = prompt_len - 1
        prev_token = tokens[0, prompt_len - 1].reshape(1).to(torch.int32)
        finished = False
        while cur_pos < total_len - 1 and not finished:
            start_time = time.time()
            self.decode_layer.show_hands_no_mtp(prev_token, ar_steps)
            elapsed = time.time() - start_time

            acc = self.decode_layer.ar_accepted_tokens_no_mtp(0).cpu()
            n_tokens = int(acc[0].item())
            emitted = acc[1 : 1 + n_tokens].tolist()
            per_step_time = elapsed / max(1, n_tokens)

            last_tok = int(prev_token[0].item())
            for tok in emitted:
                if cur_pos + 1 >= total_len:
                    break
                tokens[0, cur_pos + 1] = tok
                cur_pos += 1
                last_tok = tok
                if cur_pos >= prompt_len and print_log:
                    print(
                        self.tokenizer.decode([tok], skip_special_tokens=True),
                        end="",
                        flush=True,
                    )
                time_list.append(per_step_time)
                if tok in self.stop_token_ids:
                    finished = True
                    break
            prev_token = torch.tensor([last_tok], dtype=torch.int32, device=self.default_device)

        if print_log:
            print("\n")
            logger.info(f"--Number of tokens generated: {len(time_list)}")
            if time_list:
                total_t = sum(time_list)
                tps = len(time_list) / total_t if total_t > 0 else 0
                logger.info(f"--Effective TPS (AR, ar_steps={ar_steps}): {tps:.2f} tokens/s")
            print("\n")

        self.decode_layer.reset_sequence()

        completion_tokens = []
        for _, toks in enumerate(tokens.tolist()):
            toks = toks[prompt_len : prompt_len + self.max_new_tokens]
            stop_idx = len(toks)
            for i, tok in enumerate(toks):
                if tok == -1 or tok in self.stop_token_ids:
                    stop_idx = i
                    break
            toks = toks[:stop_idx]
            completion_tokens.append(toks)
        decoded_tokens = self.tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        return f"{decoded_tokens[0]}\n" if decoded_tokens else "", time_list, prompt_len

    def _generate_with_mtp(
        self,
        prompt: str,
        print_log: bool = True,
        prompt_tokens: list[int] | None = None,
    ) -> tuple[str, list[float], list[int], int]:
        """Generation with MTP (Multi-Token Prediction) speculative decoding."""
        if prompt_tokens is None:
            prompt_tokens = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                return_dict=False,
            )

        max_seq_len = self.config.max_seq_len
        prompt_len = len(prompt_tokens)
        total_len = min(max_seq_len, self.max_new_tokens + prompt_len)

        tokens = torch.full(
            (self.batch_size, total_len), -1, dtype=torch.long, device=self.default_device
        )
        tokens[0, :prompt_len] = torch.tensor(
            prompt_tokens, dtype=torch.long, device=self.default_device
        )

        prefill_time_list = []
        cur_pos = 0

        while cur_pos < prompt_len - 1:
            draft_end = min(cur_pos + self.mtp_seq_len, prompt_len)
            draft_tokens = tokens[0, cur_pos:draft_end].clone()
            actual_token_count = draft_tokens.shape[0]

            if actual_token_count < self.mtp_seq_len:
                pad_token = draft_tokens[-1].item()
                padding = torch.full(
                    (self.mtp_seq_len - actual_token_count,),
                    pad_token,
                    dtype=torch.long,
                    device=self.default_device,
                )
                draft_tokens = torch.cat([draft_tokens, padding])

            draft_tokens = draft_tokens.reshape(1, self.mtp_seq_len).to(torch.int32)

            mtp_extra_pos = cur_pos + self.mtp_seq_len
            if mtp_extra_pos < prompt_len:
                mtp_extra_token = int(tokens[0, mtp_extra_pos].item())
            else:
                mtp_extra_token = int(tokens[0, draft_end - 1].item())
            self.decode_layer.set_prefill_mtp_extra_token(mtp_extra_token)

            self.decode_layer.set_prefill_valid_tokens(actual_token_count)

            start_time = time.time()
            self.decode_layer.forward(draft_tokens, with_mtp=True)
            end_time = time.time()
            prefill_time_list.append(end_time - start_time)

            cur_pos += actual_token_count

        cur_pos = prompt_len - 1
        self.set_cur_pos(prompt_len - 1)

        self.decode_layer.set_prefill_valid_tokens(0)

        ar_steps = max(1, min(1024, int(os.environ.get("GLM5_AR_N", "8"))))
        return self._decode_ar(tokens, cur_pos, prompt_len, total_len, ar_steps, print_log)

    def _decode_ar(
        self,
        tokens: torch.Tensor,
        cur_pos: int,
        prompt_len: int,
        total_len: int,
        ar_steps: int,
        print_log: bool,
    ) -> tuple[str, list[float], list[int], int]:
        """MTP decode (unified single-op, unified-style)."""
        decode_time_list: list[float] = []
        decode_accepted_counts: list[int] = []

        last_token = tokens[0, prompt_len - 1].item()
        prev_draft = torch.full(
            (1, self.mtp_seq_len),
            last_token,
            dtype=torch.int32,
            device=self.default_device,
        )

        finished = False
        while cur_pos < total_len - 1 and not finished:
            start_time = time.time()
            self.decode_layer.show_hands(prev_draft, ar_steps)
            elapsed = time.time() - start_time

            acc = self.decode_layer.ar_accepted_tokens(0).cpu()
            num = self.decode_layer.ar_num_accepted(0).cpu()
            n_tokens = int(acc[0].item())
            n_steps = int(num[0].item())
            emitted = acc[1 : 1 + n_tokens].tolist()
            per_step = num[1 : 1 + n_steps].tolist()
            next_prev_draft = self.decode_layer.get_next_draft_tokens(0).reshape(
                1, self.mtp_seq_len
            )

            per_step_time = elapsed / max(1, len(per_step))
            offset = 0
            for na in per_step:
                step_emit = emitted[offset : offset + na]
                offset += na
                for tok in step_emit:
                    if cur_pos + 1 >= total_len:
                        break
                    tokens[0, cur_pos + 1] = tok
                    cur_pos += 1
                    if cur_pos >= prompt_len and print_log:
                        print(
                            self.tokenizer.decode([tok], skip_special_tokens=True),
                            end="",
                            flush=True,
                        )
                    if tok in self.stop_token_ids:
                        finished = True
                        break
                decode_time_list.append(per_step_time)
                decode_accepted_counts.append(na)
                if finished or cur_pos >= total_len - 1:
                    break
            prev_draft = next_prev_draft

        if print_log:
            print("\n")
            total_tokens = sum(decode_accepted_counts)
            logger.info(f"--Number of forward calls (decode): {len(decode_accepted_counts)}")
            logger.info(f"--Total tokens generated: {total_tokens}")
            if decode_time_list:
                total_decode_time = sum(decode_time_list)
                effective_tps = total_tokens / total_decode_time if total_decode_time > 0 else 0
                logger.info(
                    f"--Effective TPS (AR, ar_steps={ar_steps}): {effective_tps:.2f} tokens/s"
                )
            print("\n")

        self.decode_layer.reset_sequence()

        completion_tokens = []
        for _, toks in enumerate(tokens.tolist()):
            toks = toks[prompt_len : prompt_len + self.max_new_tokens]
            toks = [t for t in toks if t != -1]
            stop_idx = len(toks)
            for i, tok in enumerate(toks):
                if tok in self.stop_token_ids:
                    stop_idx = i
                    break
            toks = toks[:stop_idx]
            completion_tokens.append(toks)

        decoded_tokens = self.tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        return (
            f"{decoded_tokens[0]}\n" if decoded_tokens else "",
            decode_time_list,
            decode_accepted_counts,
            prompt_len,
        )

    def inject_cache(
        self,
        layer_caches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        start_pos: int = 0,
        end_pos: int | None = None,
    ) -> None:
        """Inject external cache data into TileRT for P/D separation."""
        num_layers = len(layer_caches)
        if num_layers == 0:
            logger.warning("inject_cache called with empty layer_caches")
            return

        first_ki, _, _ = layer_caches[0]
        seqlen = first_ki.size(0)
        if end_pos is None:
            end_pos = start_pos + seqlen

        cache_len = end_pos - start_pos
        logger.info(f"Injecting cache: {num_layers} layers, positions [{start_pos}, {end_pos})")

        num_devices = self.decode_layer.num_devices

        for device_id in range(num_devices):
            _, caches, _, _ = self.decode_layer._get_device_result(device_id)

            for layer_id, (ki, kv, pe) in enumerate(layer_caches):
                if layer_id >= num_layers:
                    logger.warning(f"Layer index {layer_id} is out of bounds, skipping.")
                    break

                base_idx = layer_id * 3

                ki_src = ki[:cache_len].to(f"cuda:{device_id}")
                kv_src = kv[:cache_len].to(f"cuda:{device_id}")
                pe_src = pe[:cache_len].to(f"cuda:{device_id}")

                for _off, _src in ((0, ki_src), (1, kv_src), (2, pe_src)):
                    _dst = caches[base_idx + _off]
                    if _dst.size(1) < end_pos:
                        continue
                    _dst[0, start_pos:end_pos, :].copy_(_src)

            torch.cuda.synchronize(device_id)

        logger.info(f"Cache injection completed for {num_devices} devices")

    def set_cur_pos(self, cur_pos: int) -> None:
        """Set the current position for RoPE in C++ backend."""
        if self.with_mtp:
            num_devices = self.decode_layer.num_devices
            for device_id in range(num_devices):
                intermediates, _, _, _ = self.decode_layer._get_device_result(device_id)
                cur_pos_tensor = intermediates[Idx.CUR_POS]
                cur_pos_tensor.fill_(cur_pos)
                torch.cuda.synchronize(device_id)
        else:
            torch.ops.tilert.dsa_show_hands_set_cur_pos_glm5(cur_pos)
            logger.info(f"Set cur_pos to {cur_pos}")

    def inject_last_hidden_state(self, last_hidden_state: torch.Tensor) -> None:
        """Inject the last hidden state for MTP mode."""
        if not self.with_mtp:
            logger.warning("inject_last_hidden_state called but with_mtp is False, skipping")
            return

        if last_hidden_state.dim() == 1:
            last_hidden_state = last_hidden_state.unsqueeze(0)

        num_devices = self.decode_layer.num_devices
        for device_id in range(num_devices):
            intermediates, _, _, _ = self.decode_layer._get_device_result(device_id)
            lhs_tensor = intermediates[Idx.LAST_HIDDEN_STATES]
            lhs_src = last_hidden_state.to(f"cuda:{device_id}")
            lhs_tensor[0, 0, :].copy_(lhs_src.squeeze(0))
            torch.cuda.synchronize(device_id)

        logger.info(f"Injected last_hidden_state to {num_devices} devices")
