"""GLM-5.2 Show Hands: the TP8 e2e Python wrapper (8x MI350X, one process)."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import torch
from safetensors import safe_open

from tilert import logger
from tilert.models.glm_5_2_rocm.checkpoint_config import (
    CONVERTER_VERSION,
    load_hf_config,
    sha256_file,
    validate_hf_config,
)
from tilert.models.glm_5_2_rocm.model_args import KIND_MOE_SHARED, ModelArgsGlm52, layer_kind
from tilert.models.glm_5_2_rocm.ops import (
    index_collective,
    llm_preprocess,
    top1_allreduce,
    top_p,
    unprojo_allreduce,
)
from tilert.models.glm_5_2_rocm.ops.rmsnorm_head_proj import TOP1_WS_WORDS
from tilert.models.glm_5_2_rocm.weight_converter import (
    fp8_ki_enabled,
    fp8_kv_enabled,
    load_rank_params,
    random_rank_params,
)

__all__ = ["Glm52ShowHands", "Idx", "TEMP_VARS_SIZE"]
HIDDEN = 6144
ROPE_DIM = 64
Q_LORA = 2048
KV_LORA = 512
KV_FP8_ROW = KV_LORA + KV_LORA // 128 * 4
NOPE = 192
V_HEAD = 256
INDEX_HEADS = 32
INDEX_DIM = 128
TOPK = 2048
HIST_ROW = 520 + 2 * 64 * 256 + 256 * 256 + 257
HIST_MAX_SAMPLES = 8
TIE_PAIRS = 4096 * 64
RADIX = 256
MOE_SLOTS = 9
TOP_K = 8
VOCAB_SHARD = 19360
NUM_PES = 8
_lease_lock = threading.Lock()
_lease_owner: weakref.ReferenceType | None = None


def _lease_holder() -> Glm52ShowHands | None:
    return _lease_owner() if _lease_owner is not None else None


class Idx:
    """Temp-var indices -- must mirror the engine's own temp table."""

    X = 0
    ROPE_FREQS = 1
    Q_PE = 2
    Q_NOPE = 3
    FLASH_ACC = 4
    FLASH_MAX = 5
    FLASH_SUM = 6
    UNPROJ_O = 7
    EXP_OUT = 8
    NORM_HIDDEN = 9
    SCORES = 10
    MOE_PROBS = 11
    MOE_IDX = 12
    HIDDEN_MID = 13
    DENSE_MID = 14
    LOGITS = 15
    NORM_OUT = 16
    TOKEN_OUT = 17
    PROB_OUT = 18
    TOPP_SEND = 19
    TOKEN_ID = 20
    CUR_POS = 21
    SAMPLING_SEED = 22
    SAMPLING_POS = 23
    AR_ACC = 24
    AR_NUM = 25
    IDX_SCORES = 26
    IDX_IQ_RT = 27
    IDX_LOGITS = 28
    IDX_HIST = 29
    IDX_TIE = 30
    IDX_SELECTS = 31
    SYM_ATTN = 32
    SYM_ATTN_TAB = 33
    SYM_FFN = 34
    SYM_FFN_TAB = 35
    SYM_TOP1 = 36
    SYM_TOP1_TAB = 37
    SYM_TOPP = 38
    SYM_TOPP_TAB = 39
    XFER_BUF = 40
    XFER_TAB = 41
    MTP_EH = 42
    MTP_HIDDEN = 43
    MTP_TOKENS = 44
    DRAFT_TOKENS = 45
    NEXT_DRAFT = 46
    NUM_ACCEPTED = 47
    LAST_TOKEN = 48
    LAST_HIDDEN = 49
    LAYER_TRACE = 50
    MTP_CUR_POS = 51
    SCORE_LINES = 52
    MOE_FLAGS = 53
    MID_PAIRS = 54
    Q_PAIRS = 55
    KV_PAIRS = 56
    PE_PAIRS = 57
    M1_PAIRS = 58
    M56_PAIRS = 59
    M56_FLAGS = 60
    PROJ_PAIRS = 61
    M5_TRIPLE_SEN = 62
    MLA_AR_QLINES = 63
    MLA_AR_KVNEW = 64
    MLA_AR_PENEW = 65
    MLA_MOE_HLINES = 66
    SS_Q_PAIRS = 67
    SS_KI_PAIRS = 68
    SS_IQ_PAIRS = 69
    IDX_IQ_RT8 = 70
    IDX_IQ_SCALE = 71
    TOP1_WS = 72


TEMP_VARS_SIZE = 73


def validate_temp_vars_layout() -> None:
    got = int(torch.ops.tilert.glm52_temp_vars_size())
    if got != TEMP_VARS_SIZE:
        raise RuntimeError(
            f"temp-var layout drift: the loaded library expects {got} slots, this package builds {TEMP_VARS_SIZE}; the library and the python package are from different builds."
        )


class Glm52ShowHands:
    """TP8 show-hands wrapper: weights, temps, caches, and the engine calls."""

    def __init__(
        self,
        model_args: ModelArgsGlm52 | None = None,
        temperature: float = 1.0,
        top_p_val: float = 0.95,
        sampling_seed: int = 42,
        use_topp: bool = False,
        max_seq_len: int | None = None,
        n_layers: int | None = None,
        num_mtp: int | None = None,
    ) -> None:
        self.args = dataclasses.replace(model_args) if model_args else ModelArgsGlm52()
        if max_seq_len is not None:
            self.args.max_seq_len = max_seq_len
        self.n_layers = self.args.n_layers if n_layers is None else n_layers
        self.num_mtp = self.args.num_mtp if num_mtp is None else num_mtp
        if self.num_mtp not in (0, 1, 3):
            raise ValueError(
                f"num_mtp must be 0 (plain), 1 (seq-2 verify) or 3 (seq-4 verify + chained drafts); got {self.num_mtp}"
            )
        if (
            isinstance(self.n_layers, bool)
            or not isinstance(self.n_layers, int)
            or (not 1 <= self.n_layers <= self.args.n_layers)
        ):
            raise ValueError(
                f"n_layers must be an int in [1, {self.args.n_layers}] (got {self.n_layers!r})"
            )
        if not 0.0 < top_p_val <= 1.0:
            raise ValueError(f"top_p must be in (0, 1] (got {top_p_val})")
        if not (math.isfinite(temperature) and temperature > 0.0):
            raise ValueError(f"temperature must be finite and > 0 (got {temperature})")
        self.temperature = temperature
        self.top_p = top_p_val
        self.sampling_seed = sampling_seed
        self.use_topp = use_topp
        self.npes = self.args.num_devices
        self.max_samples = self.num_mtp + 1
        self._temps: list[list[torch.Tensor]] = []
        self._caches: list[list[torch.Tensor]] = []
        self._prepared = False

    def _build_temp_vars(self, rank: int) -> list[torch.Tensor]:
        a = self.args
        dev = f"cuda:{rank}"
        S = self.max_samples
        L = a.max_seq_len
        heads = a.local_heads
        bf16 = {"dtype": torch.bfloat16, "device": dev}
        f32 = {"dtype": torch.float32, "device": dev}
        i32 = {"dtype": torch.int32, "device": dev}
        i64 = {"dtype": torch.int64, "device": dev}
        u8 = {"dtype": torch.uint8, "device": dev}
        splits = self.num_mtp + 1
        from tilert.models.glm_5_2_rocm.ops.flash_sparse_mla import split_tile_n

        n_splits = TOPK // split_tile_n()
        t: list[torch.Tensor] = [torch.empty(0)] * TEMP_VARS_SIZE
        t[Idx.X] = torch.zeros(S, HIDDEN, **bf16)
        t[Idx.ROPE_FREQS] = torch.zeros(S, ROPE_DIM, **f32)
        t[Idx.Q_PE] = torch.zeros(S, heads * ROPE_DIM, **bf16)
        t[Idx.Q_NOPE] = torch.zeros(S, heads * KV_LORA, **bf16)
        t[Idx.FLASH_ACC] = torch.zeros(S, heads, n_splits, KV_LORA, **f32)
        t[Idx.FLASH_MAX] = torch.zeros(S, heads, n_splits, **f32)
        t[Idx.FLASH_SUM] = torch.zeros(S, heads, n_splits, **f32)
        t[Idx.UNPROJ_O] = torch.zeros(S, HIDDEN, **bf16)
        t[Idx.EXP_OUT] = torch.zeros(S, HIDDEN, **bf16)
        t[Idx.NORM_HIDDEN] = torch.zeros(S, HIDDEN, **bf16)
        t[Idx.SCORES] = torch.zeros(S, a.n_routed_experts, **f32)
        t[Idx.LAYER_TRACE] = torch.zeros(self.n_layers + 1, S, HIDDEN, **bf16)
        t[Idx.MTP_CUR_POS] = torch.zeros(a.max_batch_size, **i32)
        t[Idx.SCORE_LINES] = torch.zeros(S, 32, 32, **i32)
        t[Idx.MOE_FLAGS] = torch.zeros(512, **i32)
        t[Idx.MID_PAIRS] = torch.zeros(S, MOE_SLOTS, 256, **i32)
        t[Idx.Q_PAIRS] = torch.zeros(S, Q_LORA // 2, 2, **i32)
        t[Idx.KV_PAIRS] = torch.zeros(S, KV_LORA // 2, 2, **i32)
        t[Idx.PE_PAIRS] = torch.zeros(S, ROPE_DIM // 2, 2, **i32)
        t[Idx.M1_PAIRS] = torch.zeros(S, heads * 256 // 2, 2, **i32)
        t[Idx.M56_PAIRS] = torch.zeros(8, heads, KV_LORA, **i32)
        t[Idx.M56_FLAGS] = torch.zeros(16 * heads, **i32)
        t[Idx.PROJ_PAIRS] = torch.zeros(heads * 16 * 4 * 16 * (1 + 8), **i32)
        t[Idx.M5_TRIPLE_SEN] = torch.zeros(2 * 8 * (TOPK // 64), **i32)
        t[Idx.MLA_AR_QLINES] = torch.zeros(heads * (KV_LORA + ROPE_DIM) // 16 * 4 * 16, **i32)
        t[Idx.MLA_AR_KVNEW] = torch.zeros(4 * KV_LORA * 2, **i32)
        t[Idx.MLA_AR_PENEW] = torch.zeros(4 * ROPE_DIM * 2, **i32)
        t[Idx.MLA_MOE_HLINES] = torch.zeros(4 * 256 * 16, **i32)
        t[Idx.SS_Q_PAIRS] = torch.zeros(S, Q_LORA // 2, 2, **i32)
        t[Idx.SS_KI_PAIRS] = torch.zeros(S, INDEX_DIM // 2, 2, **i32)
        t[Idx.SS_IQ_PAIRS] = torch.zeros(S, INDEX_HEADS * INDEX_DIM // 2, 2, **i32)
        t[Idx.MOE_PROBS] = torch.zeros(S, TOP_K, **f32)
        t[Idx.MOE_IDX] = torch.zeros(S, TOP_K, **i32)
        t[Idx.HIDDEN_MID] = torch.zeros(S, MOE_SLOTS, a.moe_inter_shard, **bf16)
        t[Idx.DENSE_MID] = torch.zeros(S, a.dense_inter_shard, **bf16)
        t[Idx.LOGITS] = torch.zeros(S, VOCAB_SHARD, **f32)
        t[Idx.NORM_OUT] = torch.zeros(S, HIDDEN, **bf16)
        t[Idx.TOKEN_OUT] = torch.zeros(S, **i32)
        t[Idx.PROB_OUT] = torch.zeros(S, **f32)
        t[Idx.TOPP_SEND] = torch.zeros(S, top_p.SEND_BYTES, **u8)
        t[Idx.TOKEN_ID] = torch.zeros(a.max_batch_size, **i32)
        t[Idx.CUR_POS] = torch.zeros(a.max_batch_size, **i32)
        t[Idx.SAMPLING_SEED] = torch.full((S,), self.sampling_seed, **i64)
        t[Idx.SAMPLING_POS] = torch.zeros(S, **i64)
        t[Idx.AR_ACC] = torch.zeros(a.max_batch_size, L + 1, **i32)
        t[Idx.AR_NUM] = torch.zeros(a.max_batch_size, L + 1, **i32)
        t[Idx.IDX_SCORES] = torch.zeros(S, INDEX_HEADS, **bf16)
        t[Idx.IDX_IQ_RT] = torch.zeros(S, INDEX_HEADS * INDEX_DIM, **bf16)
        t[Idx.IDX_IQ_RT8] = torch.zeros(S, INDEX_HEADS * INDEX_DIM, **u8)
        t[Idx.IDX_IQ_SCALE] = torch.zeros(S, INDEX_HEADS, **f32)
        t[Idx.IDX_LOGITS] = torch.zeros(S, L, **f32)
        t[Idx.IDX_HIST] = torch.zeros(HIST_MAX_SAMPLES * HIST_ROW, **i32)
        t[Idx.IDX_TIE] = torch.zeros(S * 2 * TIE_PAIRS, **i32)
        t[Idx.IDX_SELECTS] = torch.zeros(S, TOPK, **i32)
        t[Idx.SYM_ATTN] = unprojo_allreduce.sym_buffer(S, dev)
        t[Idx.SYM_ATTN_TAB] = torch.zeros(NUM_PES, **i64)
        t[Idx.SYM_FFN] = unprojo_allreduce.sym_buffer(S, dev)
        t[Idx.SYM_FFN_TAB] = torch.zeros(NUM_PES, **i64)
        t[Idx.SYM_TOP1] = top1_allreduce.sym_buffer(S, dev)
        t[Idx.SYM_TOP1_TAB] = torch.zeros(NUM_PES, **i64)
        t[Idx.SYM_TOPP] = top_p.sym_buffer(S, dev)
        t[Idx.SYM_TOPP_TAB] = torch.zeros(NUM_PES, **i64)
        t[Idx.XFER_BUF] = torch.zeros(index_collective.xfer_buf_bytes(S, TOPK) // 4, **i32)
        t[Idx.XFER_TAB] = torch.zeros(NUM_PES, **i64)
        t[Idx.MTP_EH] = torch.zeros(S, HIDDEN, **bf16)
        t[Idx.MTP_HIDDEN] = torch.zeros(S, HIDDEN, **bf16)
        t[Idx.MTP_TOKENS] = torch.zeros(S, **i32)
        t[Idx.DRAFT_TOKENS] = torch.zeros(a.max_batch_size * S, **i32)
        t[Idx.NEXT_DRAFT] = torch.zeros(a.max_batch_size * S, **i32)
        t[Idx.NUM_ACCEPTED] = torch.zeros(a.max_batch_size, **i32)
        t[Idx.LAST_TOKEN] = torch.zeros(a.max_batch_size, **i32)
        ws_words = TOP1_WS_WORDS
        t[Idx.TOP1_WS] = torch.zeros(ws_words, dtype=torch.int64, device=dev)
        t[Idx.LAST_HIDDEN] = torch.zeros(a.max_batch_size, HIDDEN, **bf16)
        assert all((x.numel() > 0 or i == 0 for i, x in enumerate(t))), "temp gap"
        del splits
        return t

    def _build_caches(self, rank: int) -> list[torch.Tensor]:
        a = self.args
        dev = f"cuda:{rank}"
        L = a.max_seq_len
        B = a.max_batch_size
        bf16 = {"dtype": torch.bfloat16, "device": dev}
        caches: list[torch.Tensor] = []
        n_extra = 1 if self.num_mtp > 0 else 0

        def kv_cache() -> torch.Tensor:
            if fp8_kv_enabled():
                return torch.zeros(B, L, KV_FP8_ROW, dtype=torch.uint8, device=dev)
            return torch.zeros(B, L, KV_LORA, **bf16)

        if rank == 0:
            for _ in range(self.n_layers):
                caches.append(kv_cache())
                caches.append(torch.zeros(B, L, ROPE_DIM, **bf16))
            n_full = sum(1 for i in range(self.n_layers) if layer_kind(i) != KIND_MOE_SHARED)
            for _ in range(n_full + n_extra):
                if fp8_ki_enabled():
                    caches.append(
                        torch.zeros(B * L * (INDEX_DIM + 4), dtype=torch.uint8, device=dev)
                    )
                else:
                    caches.append(torch.zeros(B, L, INDEX_DIM, **bf16))
        else:
            for _ in range(self.n_layers + n_extra):
                caches.append(kv_cache())
                caches.append(torch.zeros(B, L, ROPE_DIM, **bf16))
        return caches

    _SYM_PAIRS = (
        (Idx.SYM_ATTN, Idx.SYM_ATTN_TAB),
        (Idx.SYM_FFN, Idx.SYM_FFN_TAB),
        (Idx.SYM_TOP1, Idx.SYM_TOP1_TAB),
        (Idx.SYM_TOPP, Idx.SYM_TOPP_TAB),
        (Idx.XFER_BUF, Idx.XFER_TAB),
    )

    def _link_sym_tables(self) -> None:
        for buf_i, tab_i in self._SYM_PAIRS:
            ptrs = [self._temps[r][buf_i].data_ptr() for r in range(self.npes)]
            for r in range(self.npes):
                self._temps[r][tab_i].copy_(
                    torch.tensor(ptrs, dtype=torch.int64), non_blocking=False
                )

    def _acquire_lease(self) -> None:
        global _lease_owner
        with _lease_lock:
            holder = _lease_holder()
            if holder is self:
                raise RuntimeError(
                    "this Glm52ShowHands instance is already prepared; call cleanup() before loading again"
                )
            if holder is not None:
                raise RuntimeError(
                    "another Glm52ShowHands instance is live in this process (the show-hands state is a singleton); call cleanup() on it first -- this instance was not prepared and the live one is untouched"
                )
            _lease_owner = weakref.ref(self)

    def owns_state(self) -> bool:
        with _lease_lock:
            return _lease_holder() is self

    def _prepare(self, params_by_rank: list[list[torch.Tensor]]) -> None:
        assert self.owns_state(), "_prepare without the lease"
        self._params = params_by_rank
        for r in range(self.npes):
            torch.ops.tilert.glm52_prepare_rank(
                r, params_by_rank[r], self._temps[r], self._caches[r]
            )
        torch.ops.tilert.glm52_show_hands_prepare_money(
            self.npes,
            self.n_layers,
            self.args.max_seq_len,
            self.num_mtp,
            self.use_topp,
            self.temperature,
            self.top_p,
        )
        self._prepared = True

    def _alloc_state(self) -> None:
        validate_temp_vars_layout()
        if self.args.max_seq_len < TOPK:
            raise ValueError(
                f"max_seq_len ({self.args.max_seq_len}) must be >= the sparse selection budget ({TOPK}); the flash kernel rejects a cache shorter than topk"
            )
        n = int(unprojo_allreduce.enable_peer_access(self.npes))
        if n < self.npes:
            raise RuntimeError(
                f"peer access enabled on only {n} of {self.npes} devices; the collectives need all 8 GPUs peer-mapped"
            )
        self._temps = []
        self._caches = []
        for r in range(self.npes):
            with torch.cuda.device(r):
                self._temps.append(self._build_temp_vars(r))
                self._caches.append(self._build_caches(r))
        self._link_sym_tables()

    def _make_freqs_cis(self) -> torch.Tensor:
        return llm_preprocess.make_freqs_cis(
            self.args.max_seq_len, theta=self.args.rope_theta, device="cpu"
        )

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def from_pretrained(self, weights_dir: str) -> None:
        self._acquire_lease()
        try:
            self._from_pretrained_impl(weights_dir)
        except BaseException:
            self.cleanup()
            raise

    def _init_random_weights_impl(self, seed: int) -> None:
        self._alloc_state()
        freqs_cis = self._make_freqs_cis()
        params = []
        for r in range(self.npes):
            with torch.cuda.device(r):
                params.append(
                    random_rank_params(
                        self.args,
                        r,
                        f"cuda:{r}",
                        self.n_layers,
                        seed=seed,
                        num_mtp=self.num_mtp,
                        freqs_cis=freqs_cis,
                    )
                )
            logger.info("rank %d: random weights packed", r)
        self._prepare(params)

    def _from_pretrained_impl(self, weights_dir: str) -> None:
        meta_path = os.path.join(weights_dir, "tilert_meta.json")
        cfg_path = os.path.join(weights_dir, "shared", "config.json")
        if not os.path.isfile(meta_path):
            raise ValueError(f"{weights_dir}: no tilert_meta.json (not a conversion?)")
        if not os.path.isfile(cfg_path):
            raise ValueError(f"{weights_dir}: no shared/config.json to validate against")
        hf_cfg = load_hf_config(cfg_path)
        validate_hf_config(
            hf_cfg,
            self.args,
            n_layers=self.n_layers,
            num_mtp=self.num_mtp,
            max_seq_len=self.args.max_seq_len,
        )
        if True:
            with open(meta_path) as f:
                meta = json.load(f)
            stamp_hint = f"; stamp it with `python -m tilert.models.glm_5_2_rocm.weight_converter --stamp-provenance --model_dir <HF checkpoint> --save_dir {weights_dir}`"
            if meta.get("model") != "glm_5_2":
                raise ValueError(
                    f"{weights_dir} holds a {meta.get('model')!r} conversion, not glm_5_2"
                    + ("" if "model" in meta else stamp_hint)
                )
            version = meta.get("converter_version")
            if version != CONVERTER_VERSION:
                raise ValueError(
                    f"{weights_dir}: converter_version {version!r} != this loader's {CONVERTER_VERSION}"
                    + stamp_hint
                )
            want_sha = meta.get("source_config_sha256")
            if not want_sha:
                raise ValueError(
                    f"{weights_dir}: tilert_meta.json carries no source_config_sha256 (no provenance)"
                    + stamp_hint
                )
            if want_sha != sha256_file(cfg_path):
                raise ValueError(
                    f"{weights_dir}: shared/config.json differs from the checkpoint config the conversion was cut from (sha256 mismatch)"
                )
            if meta.get("n_layers", self.n_layers) < self.n_layers:
                raise ValueError(
                    f"{weights_dir} holds only {meta['n_layers']} layers, this run wants {self.n_layers}"
                )
            if self.num_mtp > 0 and meta.get("num_mtp", 0) < 1:
                raise ValueError(
                    f"{weights_dir} was converted without the MTP module (rerun the converter with --num_mtp 1) or construct with num_mtp=0"
                )
            if not meta.get("attn_fp8_lossless", False):
                raise ValueError(
                    f"{weights_dir} holds RE-QUANTIZED Wq_b/Wkv_b tensors (128-row scales); the attention kernels read 64-row-stripe scales over the checkpoint's own fp8 bytes. Repair in place with: python -m tilert.models.glm_5_2_rocm.weight_converter --augment-attn-lossless --model_dir <HF checkpoint> --save_dir {weights_dir}"
                )
            if not meta.get("attn_tp8", False):
                raise ValueError(
                    f"{weights_dir} was converted without the attn_tp8 set, which the pure-MLA TP8 runner reads on every shared layer. Either re-run the converter or augment the existing directory with weight_converter.augment_attn_tp8"
                )
        self._alloc_state()
        with safe_open(
            os.path.join(weights_dir, "shared", "embed.safetensors"), framework="pt"
        ) as f:
            embed = f.get_tensor("embed")
        freqs_cis = self._make_freqs_cis()

        def load_one(r: int) -> list[torch.Tensor]:
            torch.cuda.set_device(r)
            out = load_rank_params(
                weights_dir,
                self.args,
                r,
                f"cuda:{r}",
                n_layers=self.n_layers,
                num_mtp=self.num_mtp,
                embed=embed,
                freqs_cis=freqs_cis,
            )
            logger.info("rank %d: weights loaded and packed", r)
            return out

        with ThreadPoolExecutor(max_workers=self.npes) as pool:
            params = list(pool.map(load_one, range(self.npes)))
        self._prepare(params)

    def _check_token(self, token_id: torch.Tensor) -> None:
        if not isinstance(token_id, torch.Tensor):
            raise TypeError(f"token_id must be an int32 [1] tensor, got {token_id!r}")
        if token_id.dtype != torch.int32 or token_id.numel() != 1:
            raise ValueError(
                f"token_id must be int32 [1], got {token_id.dtype} {tuple(token_id.shape)}"
            )
        tok = int(token_id.item())
        if not 0 <= tok < self.args.vocab_size:
            raise ValueError(f"token_id {tok} outside [0, {self.args.vocab_size})")

    def _check_steps(self, n: object) -> int:
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"step count must be an int, got {n!r}")
        if not 1 <= n <= self.args.max_seq_len:
            raise ValueError(f"step count {n} outside [1, max_seq_len = {self.args.max_seq_len}]")
        return n

    def forward(self, token_id: torch.Tensor) -> None:
        assert self._prepared, "call init_random_weights() or from_pretrained()"
        self._check_token(token_id)
        torch.ops.tilert.glm52_show_hands(token_id)

    def step(self, token_id: int) -> int:
        tok = torch.tensor([token_id], dtype=torch.int32)
        self.forward(tok)
        return int(self._temps[0][Idx.TOKEN_OUT][0].item())

    def prefill(self, token_id: torch.Tensor, next_token: int) -> None:
        assert self._prepared, "call init_random_weights() or from_pretrained()"
        self._check_token(token_id)
        if isinstance(next_token, bool) or not isinstance(next_token, int):
            raise TypeError(f"next_token must be an int, got {next_token!r}")
        if not -1 <= next_token < self.args.vocab_size:
            raise ValueError(f"next_token {next_token} outside [-1, {self.args.vocab_size})")
        torch.ops.tilert.glm52_show_hands_prefill(
            token_id, torch.tensor([next_token], dtype=torch.int32)
        )

    def decode_n(self, n: int) -> None:
        torch.ops.tilert.glm52_show_hands_decode_n(self._check_steps(n))

    def mtp_n(self, n: int) -> int:
        return int(torch.ops.tilert.glm52_show_hands_mtp_n(self._check_steps(n)))

    def seed_draft(self, token: int, draft: int) -> None:
        torch.ops.tilert.glm52_seed_draft(token, draft)

    @property
    def accepted_count(self) -> int:
        return int(self._temps[0][Idx.AR_ACC][0, 0].item())

    def accepted_tokens(self, start: int = 0, end: int | None = None) -> list[int]:
        end = self.accepted_count if end is None else end
        if end <= start:
            return []
        row = self._temps[0][Idx.AR_ACC][0, 1 + start : 1 + end]
        return row.cpu().tolist()

    @property
    def step_count(self) -> int:
        return int(self._temps[0][Idx.AR_NUM][0, 0].item())

    def accepted_step_counts(self, start: int = 0, end: int | None = None) -> list[int]:
        end = self.step_count if end is None else end
        if end <= start:
            return []
        row = self._temps[0][Idx.AR_NUM][0, 1 + start : 1 + end]
        return row.cpu().tolist()

    @property
    def token_out(self) -> int:
        return int(self._temps[0][Idx.TOKEN_OUT][0].item())

    def update_sampling(self, use_topp: bool, temperature: float, top_p_val: float) -> None:
        if not 0.0 < top_p_val <= 1.0:
            raise ValueError(f"top_p must be in (0, 1] (got {top_p_val})")
        if not (math.isfinite(temperature) and temperature > 0.0):
            raise ValueError(f"temperature must be finite and > 0 (got {temperature})")
        torch.ops.tilert.glm52_update_sampling(use_topp, temperature, top_p_val)
        self.use_topp = use_topp
        self.temperature = temperature
        self.top_p = top_p_val

    def reset_sequence(self) -> None:
        torch.ops.tilert.glm52_show_hands_reset()

    def set_cur_pos(self, cur_pos: int) -> None:
        torch.ops.tilert.glm52_show_hands_set_cur_pos(cur_pos)

    def cleanup(self) -> None:
        global _lease_owner
        self._prepared = False
        try:
            with _lease_lock:
                owns = _lease_holder() is self
            if owns:
                torch.ops.tilert.glm52_show_hands_go_home()
        finally:
            with _lease_lock:
                if _lease_holder() is self:
                    _lease_owner = None
            self._params = []
            self._temps = []
            self._caches = []

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.cleanup()
