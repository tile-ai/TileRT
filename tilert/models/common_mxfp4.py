from __future__ import annotations

import os

import torch

__all__ = [
    "E2M1_TABLE",
    "dequant_mxfp4_to_bf16",
    "marlin_pack_fp4_tiles",
    "make_bf16_prefolded_scale",
    "make_random_mxfp4_weight",
    "standard_bf16_scale_bytes",
    "build_ug_weights_mma_natural",
    "build_down_weights_mma_natural",
]
E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def dequant_mxfp4_to_bf16(
    fp4_nibbles: torch.Tensor, e8m0: torch.Tensor, block_size: int = 32
) -> torch.Tensor:
    assert fp4_nibbles.dtype == torch.uint8, fp4_nibbles.dtype
    assert e8m0.dtype == torch.uint8, e8m0.dtype
    assert fp4_nibbles.shape[-1] % block_size == 0
    assert fp4_nibbles.shape[:-1] == e8m0.shape[:-1]
    assert e8m0.shape[-1] == fp4_nibbles.shape[-1] // block_size
    table = E2M1_TABLE.to(fp4_nibbles.device)
    vals_f32 = table[fp4_nibbles.long()]
    scale_f32 = torch.pow(2.0, e8m0.to(torch.float32) - 127.0)
    scale_f32 = scale_f32.repeat_interleave(block_size, dim=-1)
    return (vals_f32 * scale_f32).to(torch.bfloat16)


def make_bf16_prefolded_scale(e8m0: torch.Tensor) -> torch.Tensor:
    assert e8m0.dtype == torch.uint8
    bits = e8m0.to(torch.int32) + 126 << 7
    return bits.to(torch.int32).bitwise_and(65535).to(torch.int16).view(torch.uint16)


def standard_bf16_scale_bytes(e8m0: torch.Tensor) -> torch.Tensor:
    bits = (e8m0.to(torch.int32) << 7).contiguous()
    return bits.view(torch.uint8).reshape(*bits.shape, 4)[..., :2].contiguous()


def _use_sm90_lop3_packing(device: torch.device) -> bool:
    if os.environ.get("TILERT_FORCE_SM90_PACKING", "") not in ("", "0"):
        return True
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    idx = device.index if device.index is not None else torch.cuda.current_device()
    major = int(torch.cuda.get_device_capability(idx)[0])
    return major < 10


def _make_mma_natural_u32(nibble_tile: torch.Tensor) -> torch.Tensor:
    assert nibble_tile.shape[-2:] == (16, 16)
    pre = nibble_tile.shape[:-2]
    device = nibble_tile.device
    nib = nibble_tile.to(torch.int64) & 15
    flat = nib.reshape(*pre, 256)
    bidx, lid = torch.meshgrid(
        torch.arange(4, device=device), torch.arange(32, device=device), indexing="ij"
    )
    m_off = (bidx & 1) * 8
    k_off = (bidx >> 1) * 8
    m = lid // 4 + m_off
    k_lo = lid % 4 * 2 + k_off
    k_hi = k_lo + 1
    idx_lo = (m * 16 + k_lo).reshape(-1)
    idx_hi = (m * 16 + k_hi).reshape(-1)
    lo_vals = flat.index_select(-1, idx_lo).reshape(*pre, 4, 32)
    hi_vals = flat.index_select(-1, idx_hi).reshape(*pre, 4, 32)
    if _use_sm90_lop3_packing(device):
        lo_sh = torch.tensor([12, 8, 4, 0], dtype=torch.int64, device=device)
        hi_sh = torch.tensor([28, 24, 20, 16], dtype=torch.int64, device=device)
        view = [1] * len(pre) + [4, 1]
        out = (lo_vals << lo_sh.view(*view)).sum(dim=-2) + (hi_vals << hi_sh.view(*view)).sum(
            dim=-2
        )
        return out.to(torch.int32).view(torch.uint32)
    bytes_ = lo_vals | hi_vals << 4
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int64, device=device)
    out = (bytes_ << shifts.view(*[1] * len(pre), 4, 1)).sum(dim=-2)
    return out.to(torch.int32).view(torch.uint32)


def build_ug_weights_mma_natural(
    gate_fp4: torch.Tensor,
    gate_e8m0: torch.Tensor,
    up_fp4: torch.Tensor,
    up_e8m0: torch.Tensor,
    dim: int,
    moe_inter_dim: int,
    *,
    sms_per_expert: int = 16,
    k_per_page: int | None = None,
) -> torch.Tensor:
    e = gate_fp4.shape[0]
    outer_iters = 2 * moe_inter_dim // 16 // 32
    n_slices = sms_per_expert * outer_iters
    rh = 16
    kp = k_per_page if k_per_page is not None else _ug_k_per_page(dim)
    assert dim % kp == 0 and kp % 256 == 0, f"bad k_per_page {kp} for dim {dim}"
    np_ = dim // kp
    sc = kp // 32
    mat_bytes_per_page = 2 * (kp // 16) * 32 * 4
    scale_bytes_per_page = 32 * sc * 2
    g_fp4 = gate_fp4.reshape(e, n_slices, rh, np_, kp)
    u_fp4 = up_fp4.reshape(e, n_slices, rh, np_, kp)
    g_e8m0 = gate_e8m0.reshape(e, n_slices, rh, np_, sc)
    u_e8m0 = up_e8m0.reshape(e, n_slices, rh, np_, sc)
    page_fp4 = torch.cat([g_fp4, u_fp4], dim=2).permute(0, 1, 3, 2, 4).contiguous()
    page_e8m0 = torch.cat([g_e8m0, u_e8m0], dim=2).permute(0, 1, 3, 2, 4).contiguous()
    nib = page_fp4.reshape(e, n_slices, np_, 2, 16, kp // 16, 16)
    nib = nib.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
    u32 = _make_mma_natural_u32(nib)
    mat_bytes = u32.view(torch.uint8).reshape(e, n_slices, np_, mat_bytes_per_page)
    scale_bytes = standard_bf16_scale_bytes(page_e8m0).reshape(
        e, n_slices, np_, scale_bytes_per_page
    )
    return torch.cat([mat_bytes, scale_bytes], dim=-1).contiguous()


def _ug_k_per_page(dim: int) -> int:
    if dim % 2048 == 0:
        return 2048
    if dim % 1792 == 0:
        return 1792
    return 1024


def build_down_weights_mma_natural(
    mat_fp4: torch.Tensor,
    mat_e8m0: torch.Tensor,
    dim: int,
    moe_inter_dim: int,
    *,
    down_num_sms: int = 128,
) -> torch.Tensor:
    e = mat_fp4.shape[0]
    dev = mat_fp4.device
    odpb = dim // down_num_sms
    pad = (odpb + 15) // 16 * 16
    n_mt = pad // 16
    n_kt = moe_inter_dim // 16
    n_sc = moe_inter_dim // 32
    mat_bytes = n_mt * n_kt * 32 * 4
    scale_bytes = pad * n_sc * 2
    mat = mat_fp4.reshape(e, down_num_sms, odpb, moe_inter_dim)
    e8 = mat_e8m0.reshape(e, down_num_sms, odpb, n_sc)
    pad_rows = pad - odpb
    if pad_rows > 0:
        mat = torch.cat(
            [
                mat,
                torch.zeros(
                    e, down_num_sms, pad_rows, moe_inter_dim, dtype=torch.uint8, device=dev
                ),
            ],
            dim=2,
        )
        e8 = torch.cat(
            [e8, torch.full((e, down_num_sms, pad_rows, n_sc), 127, dtype=torch.uint8, device=dev)],
            dim=2,
        )
    nib = mat.reshape(e, down_num_sms, n_mt, 16, n_kt, 16)
    nib = nib.permute(0, 1, 2, 4, 3, 5).contiguous()
    u32 = _make_mma_natural_u32(nib)
    mat_b = u32.view(torch.uint8).reshape(e, down_num_sms, mat_bytes)
    scale_b = standard_bf16_scale_bytes(e8).reshape(e, down_num_sms, scale_bytes)
    return torch.cat([mat_b, scale_b], dim=-1).contiguous()


SF_VEC_NVFP4 = 16


def build_ug_weights_mma_natural_nvfp4(
    gate_fp4: torch.Tensor,
    gate_sf: torch.Tensor,
    up_fp4: torch.Tensor,
    up_sf: torch.Tensor,
    dim: int,
    moe_inter_dim: int,
    *,
    sms_per_expert: int = 16,
) -> torch.Tensor:
    e = gate_fp4.shape[0]
    outer_iters = 2 * moe_inter_dim // 16 // 32
    n_slices = sms_per_expert * outer_iters
    rh = 16
    kp = _ug_k_per_page(dim)
    np_ = dim // kp
    sc = kp // SF_VEC_NVFP4
    mat_bytes_per_page = 2 * (kp // 16) * 32 * 4
    scale_bytes_per_page = 32 * sc
    g_fp4 = gate_fp4.reshape(e, n_slices, rh, np_, kp)
    u_fp4 = up_fp4.reshape(e, n_slices, rh, np_, kp)
    g_sf = gate_sf.reshape(e, n_slices, rh, np_, sc)
    u_sf = up_sf.reshape(e, n_slices, rh, np_, sc)
    page_fp4 = torch.cat([g_fp4, u_fp4], dim=2).permute(0, 1, 3, 2, 4).contiguous()
    page_sf = torch.cat([g_sf, u_sf], dim=2).permute(0, 1, 3, 2, 4).contiguous()
    nib = page_fp4.reshape(e, n_slices, np_, 2, 16, kp // 16, 16)
    nib = nib.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
    u32 = _make_mma_natural_u32(nib)
    mat_bytes = u32.view(torch.uint8).reshape(e, n_slices, np_, mat_bytes_per_page)
    scale_bytes = page_sf.reshape(e, n_slices, np_, scale_bytes_per_page)
    return torch.cat([mat_bytes, scale_bytes], dim=-1).contiguous()


def build_down_weights_mma_natural_nvfp4(
    mat_fp4: torch.Tensor,
    mat_sf: torch.Tensor,
    dim: int,
    moe_inter_dim: int,
    *,
    down_num_sms: int = 128,
) -> torch.Tensor:
    e = mat_fp4.shape[0]
    dev = mat_fp4.device
    odpb = dim // down_num_sms
    pad = (odpb + 15) // 16 * 16
    n_mt = pad // 16
    n_kt = moe_inter_dim // 16
    n_sc = moe_inter_dim // SF_VEC_NVFP4
    mat_bytes = n_mt * n_kt * 32 * 4
    scale_bytes = pad * n_sc
    mat = mat_fp4.reshape(e, down_num_sms, odpb, moe_inter_dim)
    sf = mat_sf.reshape(e, down_num_sms, odpb, n_sc)
    pad_rows = pad - odpb
    if pad_rows > 0:
        mat = torch.cat(
            [
                mat,
                torch.zeros(
                    e, down_num_sms, pad_rows, moe_inter_dim, dtype=torch.uint8, device=dev
                ),
            ],
            dim=2,
        )
        sf = torch.cat(
            [sf, torch.zeros((e, down_num_sms, pad_rows, n_sc), dtype=torch.uint8, device=dev)],
            dim=2,
        )
    nib = mat.reshape(e, down_num_sms, n_mt, 16, n_kt, 16)
    nib = nib.permute(0, 1, 2, 4, 3, 5).contiguous()
    u32 = _make_mma_natural_u32(nib)
    mat_b = u32.view(torch.uint8).reshape(e, down_num_sms, mat_bytes)
    scale_b = sf.reshape(e, down_num_sms, scale_bytes)
    return torch.cat([mat_b, scale_b], dim=-1).contiguous()


_UG_KPAGE_NVFP4 = 256
_UG_KTILE_NVFP4 = 64
_UG_SFA_ATOM_NVFP4 = 512


def _sfa_perm_nvfp4(device: torch.device) -> torch.Tensor:
    return torch.tensor([m % 32 * 4 + m // 32 for m in range(128)], dtype=torch.long, device=device)


def _interleave_mat_nvfp4(nib: torch.Tensor) -> torch.Tensor:
    e, m, k = nib.shape
    assert m % 8 == 0 and k % 32 == 0
    packed = (nib[:, :, 0::2] | nib[:, :, 1::2] << 4).to(torch.uint8)
    ku128 = k // 32
    inter = packed.reshape(e, m // 8, 8, ku128, 16).permute(0, 1, 3, 2, 4).contiguous()
    return inter.reshape(e, -1)


def _sfa_atoms_nvfp4(sb: torch.Tensor) -> torch.Tensor:
    e, m, n_blk = sb.shape
    assert m == 128 and n_blk % 4 == 0
    n_atom = n_blk // 4
    u32 = sb.to(torch.int32).reshape(e, 128, n_atom, 4)
    packed = u32[..., 0] | u32[..., 1] << 8 | u32[..., 2] << 16 | u32[..., 3] << 24
    atom = torch.zeros(e, n_atom, 128, dtype=torch.int32, device=sb.device)
    atom[:, :, _sfa_perm_nvfp4(sb.device)] = packed.permute(0, 2, 1).contiguous()
    return atom.reshape(e, -1).view(torch.uint8)


def build_ug_weights_utcmma_nvfp4(
    gate_nib: torch.Tensor,
    gate_sf: torch.Tensor,
    up_nib: torch.Tensor,
    up_sf: torch.Tensor,
    dim: int,
    moe_inter_dim: int,
) -> torch.Tensor:
    e = gate_nib.shape[0]
    ug_mtiles = moe_inter_dim // 128
    n_pages = dim // _UG_KPAGE_NVFP4
    ug_ktiles = _UG_KPAGE_NVFP4 // _UG_KTILE_NVFP4
    mat_b = 128 * _UG_KPAGE_NVFP4 // 2
    sfa_b = _UG_SFA_ATOM_NVFP4 * ug_ktiles
    page_b = 2 * (mat_b + sfa_b)
    sc_per_page = _UG_KPAGE_NVFP4 // SF_VEC_NVFP4
    out = torch.zeros(e, ug_mtiles, n_pages, page_b, dtype=torch.uint8, device=gate_nib.device)
    for h in range(ug_mtiles):
        r = slice(h * 128, h * 128 + 128)
        for p in range(n_pages):
            ks = slice(p * _UG_KPAGE_NVFP4, p * _UG_KPAGE_NVFP4 + _UG_KPAGE_NVFP4)
            ss = slice(p * sc_per_page, p * sc_per_page + sc_per_page)
            out[:, h, p, 0:mat_b] = _interleave_mat_nvfp4(gate_nib[:, r, ks])
            out[:, h, p, mat_b : 2 * mat_b] = _interleave_mat_nvfp4(up_nib[:, r, ks])
            out[:, h, p, 2 * mat_b : 2 * mat_b + sfa_b] = _sfa_atoms_nvfp4(gate_sf[:, r, ss])
            out[:, h, p, 2 * mat_b + sfa_b : page_b] = _sfa_atoms_nvfp4(up_sf[:, r, ss])
    return out.reshape(e, -1).contiguous()


def build_down_weights_utcmma_nvfp4(
    mat_nib: torch.Tensor,
    mat_sf: torch.Tensor,
    dim: int,
    moe_inter_dim: int,
    *,
    down_num_sms: int = 128,
) -> torch.Tensor:
    e = mat_nib.shape[0]
    dev = mat_nib.device
    odpb = dim // down_num_sms
    n_sc = moe_inter_dim // SF_VEC_NVFP4
    mat_b = odpb * moe_inter_dim // 2
    sfa_b = _UG_SFA_ATOM_NVFP4 * (moe_inter_dim // _UG_KTILE_NVFP4)
    per = mat_b + sfa_b
    out = torch.zeros(e, down_num_sms, per, dtype=torch.uint8, device=dev)
    for c in range(down_num_sms):
        r = slice(c * odpb, c * odpb + odpb)
        sb_pad = torch.zeros(e, 128, n_sc, dtype=torch.uint8, device=dev)
        sb_pad[:, :odpb] = mat_sf[:, r, :]
        out[:, c, 0:mat_b] = _interleave_mat_nvfp4(mat_nib[:, r, :])
        out[:, c, mat_b:per] = _sfa_atoms_nvfp4(sb_pad)
    return out.contiguous()


def marlin_pack_fp4_tiles(fp4_nibbles: torch.Tensor) -> torch.Tensor:
    assert fp4_nibbles.dtype == torch.uint8
    assert fp4_nibbles.shape[-2] % 16 == 0
    assert fp4_nibbles.shape[-1] % 16 == 0
    M, K = fp4_nibbles.shape[-2:]
    pre = fp4_nibbles.shape[:-2]
    tiles = fp4_nibbles.reshape(*pre, M // 16, 16, K // 16, 16)
    tiles = tiles.permute(*range(len(pre)), -4, -2, -3, -1).contiguous()
    device = fp4_nibbles.device
    lid = torch.arange(32, device=device)
    row_t = (lid // 4).to(torch.long)
    col_t = (lid % 4 * 2).to(torch.long)

    def _gather(row_offset: int, col_offset: int) -> torch.Tensor:
        rows = (row_t + row_offset).view(32)
        cols = (col_t + col_offset).view(32)
        return tiles[..., rows, cols]

    nib0 = _gather(0, 0)
    nib1 = _gather(0, 8)
    nib2 = _gather(8, 0)
    nib3 = _gather(8, 8)
    nib4 = _gather(0, 1)
    nib5 = _gather(0, 9)
    nib6 = _gather(8, 1)
    nib7 = _gather(8, 9)

    def _u32(x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.int32).bitwise_and(15)

    packed = (
        _u32(nib0) << 0
        | _u32(nib1) << 4
        | _u32(nib2) << 8
        | _u32(nib3) << 12
        | _u32(nib4) << 16
        | _u32(nib5) << 20
        | _u32(nib6) << 24
        | _u32(nib7) << 28
    )
    return packed.to(torch.int32).view(torch.uint32)


def make_random_mxfp4_weight(
    shape: tuple[int, ...],
    *,
    e8m0_min: int = 110,
    e8m0_max: int = 120,
    block_size: int = 32,
    device: str | torch.device = "cuda",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert shape[-1] % block_size == 0
    fp4 = torch.randint(0, 16, shape, dtype=torch.uint8, device=device, generator=generator)
    scale_shape = (*shape[:-1], shape[-1] // block_size)
    e8m0 = torch.randint(
        e8m0_min, e8m0_max + 1, scale_shape, dtype=torch.uint8, device=device, generator=generator
    )
    dequant = dequant_mxfp4_to_bf16(fp4, e8m0, block_size)
    return (fp4, e8m0, dequant)


def _unpack_fp4_nibbles_last(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        packed = packed.contiguous().view(torch.uint8)
    lo = packed.bitwise_and(15)
    hi = packed.bitwise_right_shift(4).bitwise_and(15)
    target_shape = list(packed.shape)
    target_shape[-1] = packed.shape[-1] * 2
    return torch.stack([lo, hi], dim=-1).reshape(target_shape).contiguous()


def _pack_fp4_nibbles_last(unpacked: torch.Tensor) -> torch.Tensor:
    assert (
        unpacked.shape[-1] % 2 == 0
    ), f"_pack_fp4_nibbles_last requires even last dim, got shape {tuple(unpacked.shape)}"
    if unpacked.dtype != torch.uint8:
        unpacked = unpacked.to(torch.uint8)
    lo = unpacked[..., 0::2]
    hi = unpacked[..., 1::2]
    return (lo | hi << 4).contiguous()
