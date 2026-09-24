"""GLM-5.2 sparse index op wrapper: indexer scores + top-2048 selection."""

import numpy as np
import torch

INDEX_HEADS = 32
INDEX_DIM = 128
TOPK = 2048
RADIX = 256
HIST_ROW = 520
HIST_LOCAL = 256 * 256
HIST_REPL = 2 * 64 * 256
HIST_FLAGS = 256 + 1
HIST_WORDS = HIST_ROW + HIST_REPL + HIST_LOCAL + HIST_FLAGS
HIST_MAX_SAMPLES = 8
PAIRS = 4096 * 64
SM_SCALE = 0.17677669529 * 0.08838834764831843
SUPPORTED_SAMPLES = (1, 2, 4, 8)


def logits_stride(cur_pos: torch.Tensor, seq_len: int) -> int:
    m = int(cur_pos.max().item()) + seq_len
    return (m + 7) // 8 * 8


def key32(v: np.ndarray) -> np.ndarray:
    b = v.astype(np.float32).view(np.uint32)
    return np.where(b & np.uint32(2147483648), ~b, b | np.uint32(2147483648))


def key8(v: np.ndarray) -> np.ndarray:
    f = np.asarray(v, dtype=np.float32)
    h = f.astype(np.float16)
    over = np.abs(h.astype(np.float32)) > np.abs(f)
    hb = h.view(np.uint16)
    hb = np.where(over, hb - np.uint16(1), hb).astype(np.uint16)
    k = np.where(hb & np.uint16(32768), ~hb, hb | np.uint16(32768)).astype(np.uint16)
    return (k >> np.uint16(8)).astype(np.uint32)


FP8_MAX = 448.0
FP8_AMAX_EPS = 0.0001


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.shape[-1] == INDEX_DIM
    xf = x.float()
    amax = xf.abs().amax(dim=-1)
    scale = amax.clamp_min(FP8_AMAX_EPS) * torch.tensor(
        1.0 / FP8_MAX, dtype=torch.float32, device=xf.device
    )
    q = (xf / scale[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (q.view(torch.uint8), scale.to(torch.float32))


def dequantize_rows(q8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q8.view(torch.float8_e4m3fn).float() * scale.float()[..., None]


class SparseIndexGlm5:
    """The scores, the histogram workspace and the selection."""

    OP_NAMES = ("glm5_sparse_index_op", "glm5_sparse_index_fp8_op", "glm5_topk_select_op")

    def __init__(self, device: str = "cuda:0", topk: int = TOPK, fp8: bool = False):
        self.device = device
        self.topk = topk
        self.fp8 = fp8

    def alloc(self, samples: int, stride: int) -> dict:
        dev = self.device
        return {
            "logits": torch.zeros(samples, stride, dtype=torch.float32, device=dev),
            "hist": torch.zeros(HIST_MAX_SAMPLES * HIST_WORDS, dtype=torch.int32, device=dev),
            "tie_pairs": torch.zeros(samples * 2 * PAIRS, dtype=torch.int32, device=dev),
            "idx": torch.zeros(samples, self.topk, dtype=torch.int32, device=dev),
        }

    @staticmethod
    def logits_golden(
        iq_rt: torch.Tensor,
        ki_cache: torch.Tensor,
        idx_scores: torch.Tensor,
        cur_pos: torch.Tensor,
        seq_len: int,
        stride: int,
    ) -> torch.Tensor:
        samples = iq_rt.shape[0]
        batch = cur_pos.numel()
        q = iq_rt.float().cpu().reshape(samples, INDEX_HEADS, INDEX_DIM)
        kc = ki_cache.float().cpu()
        w = idx_scores.float().cpu()
        out = torch.zeros(samples, stride, dtype=torch.float32)
        for b in range(batch):
            n = int(cur_pos[b]) + seq_len
            keys = kc[b, :n]
            for s in range(seq_len):
                sample = b * seq_len + s
                sc = torch.relu(q[sample] @ keys.T * SM_SCALE)
                out[sample, :n] = (sc * w[sample][:, None]).sum(dim=0)
        return out

    def select_golden(
        self, logits: torch.Tensor, cur_pos: torch.Tensor, seq_len: int
    ) -> list[dict]:
        samples = logits.shape[0]
        lg = logits.float().cpu().numpy()
        res = []
        for sample in range(samples):
            b, s = (sample // seq_len, sample % seq_len)
            bound = int(cur_pos[b]) + s + 1
            keys = key32(lg[sample, :bound])
            if bound <= self.topk:
                res.append({"identity": True, "bound": bound})
                continue
            kstar = int(np.sort(keys)[::-1][self.topk - 1])
            res.append(
                {
                    "identity": False,
                    "bound": bound,
                    "kstar": kstar,
                    "strict": set(np.nonzero(keys > kstar)[0].tolist()),
                    "equal": set(np.nonzero(keys == kstar)[0].tolist()),
                }
            )
        return res

    def tilert_scores(
        self,
        iq_rt: torch.Tensor,
        ki_cache: torch.Tensor,
        idx_scores: torch.Tensor,
        cur_pos: torch.Tensor,
        seq_len: int,
        ws: dict,
    ) -> torch.Tensor:
        torch.ops.tilert.glm5_sparse_index_op(
            iq_rt, ki_cache, idx_scores, cur_pos, ws["logits"], ws["hist"], seq_len, self.topk
        )
        return ws["logits"]

    def tilert_scores_fp8(
        self,
        iq_rt8: torch.Tensor,
        iq_scale: torch.Tensor,
        ki_cache8: torch.Tensor,
        ki_scale: torch.Tensor,
        idx_scores: torch.Tensor,
        cur_pos: torch.Tensor,
        seq_len: int,
        ws: dict,
    ) -> torch.Tensor:
        torch.ops.tilert.glm5_sparse_index_fp8_op(
            iq_rt8,
            iq_scale,
            ki_cache8,
            ki_scale,
            idx_scores,
            cur_pos,
            ws["logits"],
            ws["hist"],
            seq_len,
            self.topk,
        )
        return ws["logits"]

    def tilert_select(self, cur_pos: torch.Tensor, seq_len: int, ws: dict) -> torch.Tensor:
        torch.ops.tilert.glm5_topk_select_op(
            ws["logits"], ws["hist"], ws["tie_pairs"], cur_pos, ws["idx"], seq_len, self.topk
        )
        return ws["idx"]

    def tilert_forward(
        self,
        iq_rt: torch.Tensor,
        ki_cache: torch.Tensor,
        idx_scores: torch.Tensor,
        cur_pos: torch.Tensor,
        seq_len: int,
        ws: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        samples = iq_rt.shape[0]
        if ws is None:
            ws = self.alloc(samples, logits_stride(cur_pos, seq_len))
        if self.fp8:
            q8, qs = quantize_rows(iq_rt.view(samples, INDEX_HEADS, INDEX_DIM))
            k8, ks = quantize_rows(ki_cache)
            logits = self.tilert_scores_fp8(
                q8.view(samples, -1).contiguous(),
                qs.contiguous(),
                k8.contiguous(),
                ks.contiguous(),
                idx_scores,
                cur_pos,
                seq_len,
                ws,
            )
        else:
            logits = self.tilert_scores(iq_rt, ki_cache, idx_scores, cur_pos, seq_len, ws)
        idx = self.tilert_select(cur_pos, seq_len, ws)
        return (logits, idx, ws)
