"""GLM-5.2 fp8 kv latent cache: the 528-byte row format."""

from __future__ import annotations

import torch

KV_LORA_RANK = 512
SCALE_BLOCK_SIZE = 128
NUM_SCALE_BLOCKS = KV_LORA_RANK // SCALE_BLOCK_SIZE
FP8_MAX = 448.0
AMAX_FLOOR = 0.0001
SCALE_BYTES = NUM_SCALE_BLOCKS * 4
KV_ROW_BYTES = KV_LORA_RANK + SCALE_BYTES
__all__ = [
    "KV_LORA_RANK",
    "SCALE_BLOCK_SIZE",
    "NUM_SCALE_BLOCKS",
    "KV_ROW_BYTES",
    "quant_kv_to_fp8_blocked",
    "dequant_fp8_blocked",
    "quant_dequant",
    "pack_kv_528",
    "unpack_kv_528",
    "quant_pack_kv",
    "dequant_kv_528",
]


def quant_kv_to_fp8_blocked(
    x: torch.Tensor, block_size: int = SCALE_BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = x.shape[-1]
    assert dim % block_size == 0, f"dim {dim} not divisible by {block_size}"
    nb = dim // block_size
    xf = x.float()
    lead = xf.shape[:-1]
    xb = xf.reshape(*lead, nb, block_size)
    amax = xb.abs().amax(dim=-1).clamp(min=AMAX_FLOOR)
    scale = (amax * (1.0 / FP8_MAX)).contiguous()
    inv = 1.0 / scale
    q = (xb * inv.unsqueeze(-1)).clamp(min=-FP8_MAX, max=FP8_MAX)
    fp8 = q.reshape(*lead, dim).to(torch.float8_e4m3fn)
    return (fp8.contiguous(), scale)


def dequant_fp8_blocked(
    fp8: torch.Tensor, scale: torch.Tensor, block_size: int = SCALE_BLOCK_SIZE
) -> torch.Tensor:
    dim = fp8.shape[-1]
    nb = dim // block_size
    lead = fp8.shape[:-1]
    xb = fp8.float().reshape(*lead, nb, block_size)
    return (xb * scale.unsqueeze(-1)).reshape(*lead, dim).to(torch.bfloat16)


def quant_dequant(x: torch.Tensor, block_size: int = SCALE_BLOCK_SIZE) -> torch.Tensor:
    fp8, scale = quant_kv_to_fp8_blocked(x, block_size)
    return dequant_fp8_blocked(fp8, scale, block_size)


def pack_kv_528(fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    lead = fp8.shape[:-1]
    fp8_u8 = fp8.reshape(-1, KV_LORA_RANK).view(torch.uint8)
    scale_u8 = scale.reshape(-1, NUM_SCALE_BLOCKS).float().contiguous().view(torch.uint8)
    return torch.cat([fp8_u8, scale_u8], dim=-1).reshape(*lead, KV_ROW_BYTES).contiguous()


def unpack_kv_528(buf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lead = buf.shape[:-1]
    flat = buf.reshape(-1, KV_ROW_BYTES)
    fp8 = flat[:, :KV_LORA_RANK].contiguous().view(torch.float8_e4m3fn)
    scale = flat[:, KV_LORA_RANK:].contiguous().view(torch.float32)
    return (fp8.reshape(*lead, KV_LORA_RANK), scale.reshape(*lead, NUM_SCALE_BLOCKS))


def quant_pack_kv(kv: torch.Tensor) -> torch.Tensor:
    return pack_kv_528(*quant_kv_to_fp8_blocked(kv))


def dequant_kv_528(buf: torch.Tensor) -> torch.Tensor:
    return dequant_fp8_blocked(*unpack_kv_528(buf))
