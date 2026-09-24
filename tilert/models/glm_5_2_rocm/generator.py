"""Glm52Generator: prompt in, text out, over the TP8 show-hands chain."""

from __future__ import annotations

import json
import os
import time

import torch
from transformers import AutoTokenizer

from tilert import logger
from tilert.models.glm_5_2_rocm.end2end import Glm52ShowHands
from tilert.models.glm_5_2_rocm.model_args import ModelArgsGlm52
from tilert.tilert_init import tilert_init

__all__ = ["Glm52Generator"]
STOP_TOKENS: dict[str, int] = {
    "<|endoftext|>": 154820,
    "<|user|>": 154827,
    "<|assistant|>": 154828,
    "<|observation|>": 154829,
}
TOP_K_FIXED = 256


def _check_max_new_tokens(n: object) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"max_new_tokens must be a non-negative int (got {n!r})")
    return n


class Glm52Generator:
    """Single-process, 8-GPU generator for GLM-5.2."""

    def __init__(
        self,
        model_weights_dir: str = "",
        model_args: ModelArgsGlm52 | None = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.95,
        sampling_seed: int = 42,
        use_topp: bool = False,
        n_layers: int | None = None,
        num_mtp: int | None = None,
        max_seq_len: int | None = None,
    ) -> None:
        self.model_weights_dir = model_weights_dir
        self.max_new_tokens = _check_max_new_tokens(max_new_tokens)
        self.decode_layer = Glm52ShowHands(
            model_args=model_args,
            temperature=temperature,
            top_p_val=top_p,
            sampling_seed=sampling_seed,
            use_topp=use_topp,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            num_mtp=num_mtp,
        )
        self.config = self.decode_layer.args
        self.use_mtp = self.decode_layer.num_mtp > 0
        self._tokenizer: AutoTokenizer | None = None
        self._stop_token_ids: set[int] = set()
        self.last_completion_tokens: list[int] = []

    def _tok_dir(self) -> str:
        shared = os.path.join(self.model_weights_dir, "shared")
        if os.path.isfile(os.path.join(shared, "tokenizer_config.json")):
            return shared
        return self.model_weights_dir

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        tok_dir = self._tok_dir()
        if not os.path.isfile(os.path.join(tok_dir, "tokenizer_config.json")):
            raise FileNotFoundError(
                f"No tokenizer_config.json under {self.model_weights_dir}. Re-run the converter so the output dir is self-contained."
            )
        self._tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        eos = self._tokenizer.eos_token_id
        if eos is not None:
            self._stop_token_ids.add(eos)
        for name, known_id in STOP_TOKENS.items():
            ids = self._tokenizer.encode(name, add_special_tokens=False)
            if len(ids) == 1:
                self._stop_token_ids.add(int(ids[0]))
                continue
            added = getattr(self._tokenizer, "added_tokens_encoder", {})
            if name in added:
                self._stop_token_ids.add(int(added[name]))
                continue
            logger.warning(
                "tokenizer does not resolve stop token %r; using GLM-5.2's id %d", name, known_id
            )
            self._stop_token_ids.add(known_id)
        gc_path = os.path.join(tok_dir, "generation_config.json")
        if os.path.isfile(gc_path):
            with open(gc_path) as f:
                gc = json.load(f)
            eos_ids = gc.get("eos_token_id", [])
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            self._stop_token_ids.update(int(t) for t in eos_ids)
        logger.info(f"Stop token IDs: {sorted(self._stop_token_ids)}")

    @property
    def tokenizer(self) -> AutoTokenizer:
        self._ensure_tokenizer()
        assert self._tokenizer is not None
        return self._tokenizer

    @property
    def stop_token_ids(self) -> set[int]:
        self._ensure_tokenizer()
        return self._stop_token_ids

    def init(self) -> None:
        tilert_init()

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def from_pretrained(self) -> None:
        self.decode_layer.from_pretrained(self.model_weights_dir)

    def cleanup(self) -> None:
        self.decode_layer.cleanup()

    def update_sampling_params(
        self,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = TOP_K_FIXED,
        use_topp: bool = True,
    ) -> None:
        if isinstance(top_k, bool) or top_k != TOP_K_FIXED:
            raise ValueError(
                f"top_k is fixed at {TOP_K_FIXED} (per-rank top-256 candidate set, 2048 globally) by the top-p kernels; got {top_k!r}"
            )
        self.decode_layer.update_sampling(use_topp, temperature, top_p)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        print_log: bool = True,
        prompt_tokens: list[int] | None = None,
        max_new_tokens: int | None = None,
        raw: bool = False,
        with_mtp: bool | None = None,
    ) -> tuple[str, list[float], list[int], int]:
        use_mtp = self.use_mtp if with_mtp is None else with_mtp
        if use_mtp:
            assert self.decode_layer.num_mtp > 0, "constructed with num_mtp=0"
        self._ensure_tokenizer()
        assert self._tokenizer is not None
        n_new = (
            self.max_new_tokens if max_new_tokens is None else _check_max_new_tokens(max_new_tokens)
        )
        if prompt_tokens is None:
            if raw:
                prompt_tokens = self._tokenizer.encode(prompt)
            else:
                messages = [{"role": "user", "content": prompt}]
                res = self._tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                prompt_tokens = res["input_ids"] if hasattr(res, "keys") else res
        prompt_len = len(prompt_tokens)
        pos_limit = self.decode_layer.args.max_seq_len
        if not prompt_tokens:
            raise ValueError("empty prompt: nothing to condition the first token on")
        vocab = self.config.vocab_size
        for i, tid in enumerate(prompt_tokens):
            if isinstance(tid, bool) or not isinstance(tid, int) or (not 0 <= tid < vocab):
                raise ValueError(f"prompt token {i} = {tid!r} is not an int in [0, {vocab})")
        if prompt_len > pos_limit:
            raise ValueError(
                f"prompt of {prompt_len} tokens does not fit the {pos_limit}-token cache (max_seq_len)"
            )
        self.last_completion_tokens = []
        if n_new == 0:
            return ("", [], [], prompt_len)
        self.decode_layer.reset_sequence()
        times: list[float] = []
        t0 = time.time()
        mtp_prefill = use_mtp
        for i, tid in enumerate(prompt_tokens):
            tok = torch.tensor([tid], dtype=torch.int32)
            if mtp_prefill:
                nxt = prompt_tokens[i + 1] if i + 1 < prompt_len else -1
                self.decode_layer.prefill(tok, nxt)
            else:
                self.decode_layer.forward(tok)
        prefill_s = time.time() - t0
        base = self.decode_layer.accepted_count
        first = self.decode_layer.token_out
        completion: list[int] = [first]
        accept_lens: list[int] = []
        if print_log:
            print(self._tokenizer.decode([first], skip_special_tokens=True), end="", flush=True)
        stopped = first in self.stop_token_ids
        if use_mtp:
            self.decode_layer.seed_draft(first, first)
            ar_steps = max(1, min(1024, int(os.environ.get("GLM5_AR_N", "8"))))
            mtp_seq = self.decode_layer.num_mtp + 1
            chain_slack = max(0, self.decode_layer.num_mtp - 1)
            step_base = self.decode_layer.step_count
            steps_done = 0
            produced = 0
            while not stopped and produced < n_new - 1:
                room = pos_limit - (prompt_len + produced)
                k = min(ar_steps, (room - chain_slack) // mtp_seq)
                if k < 1:
                    logger.warning(
                        "stopping decode: position %d + %d would reach the %d-token cache limit",
                        prompt_len + produced,
                        mtp_seq + chain_slack,
                        pos_limit,
                    )
                    break
                t0 = time.time()
                got = self.decode_layer.mtp_n(k)
                dt = time.time() - t0
                new = self.decode_layer.accepted_tokens(base + produced)
                per_step = self.decode_layer.accepted_step_counts(step_base + steps_done)
                steps_done += len(per_step)
                step_dt = dt / max(1, len(per_step))
                offset = 0
                for acc in per_step:
                    step_toks = new[offset : offset + acc]
                    offset += acc
                    taken = 0
                    for tok in step_toks:
                        if len(completion) >= n_new:
                            stopped = True
                            break
                        completion.append(tok)
                        taken += 1
                        if print_log:
                            print(
                                self._tokenizer.decode([tok], skip_special_tokens=True),
                                end="",
                                flush=True,
                            )
                        if tok in self.stop_token_ids:
                            stopped = True
                            break
                    if taken > 0:
                        times.append(step_dt)
                        accept_lens.append(taken)
                    if stopped:
                        break
                produced += got
        else:
            chunk = 8
            produced = 0
            while not stopped and produced < n_new - 1:
                room = pos_limit - (prompt_len + produced)
                if room < 1:
                    logger.warning(
                        "stopping decode: position %d reached the %d-token cache limit",
                        prompt_len + produced,
                        pos_limit,
                    )
                    break
                n = min(chunk, n_new - 1 - produced, room)
                t0 = time.time()
                self.decode_layer.decode_n(n)
                dt = time.time() - t0
                new = self.decode_layer.accepted_tokens(base + produced)
                for tok in new:
                    completion.append(tok)
                    times.append(dt / n)
                    if print_log:
                        print(
                            self._tokenizer.decode([tok], skip_special_tokens=True),
                            end="",
                            flush=True,
                        )
                    if tok in self.stop_token_ids:
                        stopped = True
                        break
                produced += n
        if print_log:
            print("\n")
            n_tok = sum(accept_lens) if accept_lens else len(times)
            logger.info(
                f"--Tokens generated: {len(completion)} (prefill {prompt_len} tok in {prefill_s:.2f}s)"
            )
            if times and n_tok:
                per_tok = sum(times) / n_tok
                logger.info("==== Performance ====")
                logger.info(f"--Average time per token: {per_tok * 1000:.4f} ms")
                logger.info(f"--Effective tokens per second: {1 / per_tok:.2f}")
            if accept_lens:
                logger.info(
                    "--MTP mean accept length: %.3f over %d steps",
                    sum(accept_lens) / len(accept_lens),
                    len(accept_lens),
                )
            print("\n")
        for i, tok in enumerate(completion):
            if tok in self.stop_token_ids:
                completion = completion[:i]
                break
        self.last_completion_tokens = completion
        text = self._tokenizer.decode(completion, skip_special_tokens=True)
        return (text, times, accept_lens, prompt_len)
