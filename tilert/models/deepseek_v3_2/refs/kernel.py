import torch

try:
    import tilelang
    import tilelang.language as T

    _HAS_TILELANG = True
except ImportError:
    _HAS_TILELANG = False

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

__all__ = [
    "weight_dequant",
    "act_quant",
    "fp8_gemm",
]

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"


def _require_tilelang(fn_name: str) -> None:
    if not _HAS_TILELANG:
        raise ImportError(f"{fn_name} requires tilelang. Install with: pip install tilelang")


def _require_triton(fn_name: str) -> None:
    if not _HAS_TRITON:
        raise ImportError(f"{fn_name} requires triton. Install with: pip install triton")


if _HAS_TRITON:

    @triton.jit
    def weight_dequant_kernel(  # type: ignore
        x_ptr,
        s_ptr,
        y_ptr,
        M_Size: tl.constexpr,
        N_Size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ) -> None:
        """
        Weight dequantization kernel.

        Dequantizes weights using the provided scaling factors and stores the
            result.

        Args:
            x_ptr (tl.pointer): Pointer to the quantized weights.
            s_ptr (tl.pointer): Pointer to the scaling factors.
            y_ptr (tl.pointer): Pointer to the output buffer for dequantized
                weights.
            M (int): Number of rows in the weight matrix.
            N (int): Number of columns in the weight matrix.
            BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

        Returns:
            None
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        n_size = tl.cdiv(N_Size, BLOCK_SIZE)
        offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs = offs_m[:, None] * N_Size + offs_n[None, :]
        mask = (offs_m[:, None] < M_Size) & (offs_n[None, :] < N_Size)
        x_in = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        s_in = tl.load(s_ptr + pid_m * n_size + pid_n)
        y_out = x_in * s_in
        tl.store(y_ptr + offs, y_out, mask=mask)


def _weight_dequant_torch(
    x_in: torch.Tensor, s_in: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """Pure-PyTorch fallback for weight_dequant (multi-GPU safe).

    Used when triton is unavailable, or when the triton kernel raises at
    launch time (e.g. ``cuPointerGetAttribute`` failing on non-device-0
    GPUs during multi-device ``init_random_weights``).
    """
    M, N = x_in.shape
    y = x_in.float().reshape(M // block_size, block_size, N // block_size, block_size)
    y = y * s_in[:, None, :, None]
    return y.reshape(M, N).to(torch.get_default_dtype())


def weight_dequant(x_in: torch.Tensor, s_in: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x_in (torch.Tensor): The quantized weight tensor of shape (M, N).
        s_in (torch.Tensor): The scale tensor of shape (M//block_size,
            N//block_size).
        block_size (int, optional): The block size to use for dequantization.
            Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions
            are not 2.
    """
    assert x_in.is_contiguous() and s_in.is_contiguous(), "Input tensors must be contiguous"
    assert x_in.dim() == 2 and s_in.dim() == 2, "Input tensors must have 2 dimensions"
    if not _HAS_TRITON:
        return _weight_dequant_torch(x_in, s_in, block_size)
    M_Size, N_Size = x_in.size()
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M_Size, meta["BLOCK_SIZE"]),
        triton.cdiv(N_Size, meta["BLOCK_SIZE"]),
    )
    try:
        y_out = torch.empty_like(x_in, dtype=torch.get_default_dtype())
        weight_dequant_kernel[grid](x_in, s_in, y_out, M_Size, N_Size, BLOCK_SIZE=block_size)
    except (ValueError, RuntimeError):
        return _weight_dequant_torch(x_in, s_in, block_size)
    return y_out


if _HAS_TILELANG:
    tilelang.set_log_level("WARNING")

    _pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }

    def _fast_log2_ceil(x):  # type: ignore
        bits_x = T.reinterpret("uint32", x)
        exp_x = (bits_x >> 23) & 0xFF
        man_bits = bits_x & ((1 << 23) - 1)
        return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))

    def _fast_pow2(x):  # type: ignore
        bits_x = (x + 127) << 23
        return T.reinterpret("float32", bits_x)

    def _fast_round_scale(amax, fp8_max_inv):  # type: ignore
        return _fast_pow2(_fast_log2_ceil(amax * fp8_max_inv))

    @tilelang.jit(pass_configs=_pass_configs)
    def act_quant_kernel(  # type: ignore
        N, in_dtype=BF16, out_dtype=FP8, scale_dtype=FP32, round_scale=False  # type: ignore
    ):  # type: ignore
        M = T.symbolic("M")
        fp8_min = -448.0
        fp8_max = 448.0
        fp8_max_inv = 1 / fp8_max
        num_stages = 0 if round_scale else 2
        blk_m = 32
        group_size = 128

        @T.prim_func
        def act_quant_kernel_(  # type: ignore
            X: T.Tensor[(M, N), in_dtype],
            Y: T.Tensor[(M, N), out_dtype],
            S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
        ):  # type: ignore
            with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
                pid_m,
                pid_n,
            ):
                x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
                x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
                amax_local = T.alloc_fragment((blk_m,), scale_dtype)
                s_local = T.alloc_fragment((blk_m,), scale_dtype)
                y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
                y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

                for _ in T.Pipelined(1, num_stages=num_stages):
                    T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                    T.copy(x_shared, x_local)
                    T.reduce_absmax(x_local, amax_local, dim=1)
                    for i in T.Parallel(blk_m):
                        amax_local[i] = T.max(amax_local[i], 1e-4)
                        if round_scale:
                            s_local[i] = _fast_round_scale(amax_local[i], fp8_max_inv)
                        else:
                            s_local[i] = amax_local[i] * fp8_max_inv
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(x_local[i, j] / s_local[i], fp8_min, fp8_max)
                    for i in T.Parallel(blk_m):
                        S[pid_m * blk_m + i, pid_n] = s_local[i]
                    T.copy(y_local, y_shared)
                    T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

        return act_quant_kernel_

    @tilelang.jit(pass_configs=_pass_configs)
    def fp8_gemm_kernel(N, K, out_dtype=BF16, accum_dtype="float32"):  # type: ignore
        assert out_dtype in [BF16, "float32"]

        M = T.symbolic("M")
        group_size = 128
        block_M = 32
        block_N = 128
        block_K = 128

        @T.prim_func
        def fp8_gemm_kernel_(  # type: ignore
            A: T.Tensor[(M, K), FP8],
            B: T.Tensor[(N, K), FP8],
            C: T.Tensor[(M, N), out_dtype],
            scales_a: T.Tensor[(M, T.ceildiv(K, group_size)), FP32],
            scales_b: T.Tensor[(T.ceildiv(N, group_size), T.ceildiv(K, group_size)), FP32],
        ):  # type: ignore
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
                bx,
                by,
            ):
                A_shared = T.alloc_shared((block_M, block_K), FP8)
                B_shared = T.alloc_shared((block_N, block_K), FP8)
                C_shared = T.alloc_shared((block_M, block_N), out_dtype)
                Scale_C_shared = T.alloc_shared((block_M), FP32)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

                T.use_swizzle(panel_size=10)

                T.clear(C_local)
                T.clear(C_local_accum)
                K_iters = T.ceildiv(K, block_K)
                for k in T.Pipelined(K_iters, num_stages=4):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    Scale_B = scales_b[bx * block_N // group_size, k]
                    for i in T.Parallel(block_M):
                        Scale_C_shared[i] = scales_a[by * block_M + i, k] * Scale_B

                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]
                    T.clear(C_local)
                T.copy(C_local_accum, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return fp8_gemm_kernel_


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized.
        Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization.
        Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    _require_tilelang("act_quant")
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    N = x.size(-1)
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    kernel = act_quant_kernel(N, round_scale=scale_fmt is not None)
    kernel(x.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
    return y, s


def fp8_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor
) -> torch.Tensor:
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    _require_tilelang("fp8_gemm")
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), "Scaling factor tensors must be contiguous"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    kernel = fp8_gemm_kernel(N, K)
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c
