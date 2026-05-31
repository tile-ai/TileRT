"""Text generation script for TileRT."""

import time
from argparse import ArgumentParser
from typing import TYPE_CHECKING

import tilert

if TYPE_CHECKING:
    from tilert.models.deepseek_v3_2.generator import DSAv32Generator
    from tilert.models.glm_5.generator import GLM5Generator
from tilert.benchmark import BenchMode
from tilert.benchmark import coding_prompt as coding_bench
from tilert.benchmark import long_prompt as long_bench
from tilert.benchmark import merge_stats, print_summary_table
from tilert.benchmark import short_prompt as short_bench
from tilert.benchmark.config import get_weights_dir


def get_generator(
    model_type: str,
    max_new_tokens: int,
    temperature: float,
    model_weights_dir: str,
    with_mtp: bool,
    top_p: float = 0.9,
    top_k: int = 256,
    enable_thinking: bool = False,
    sampling_seed: int = 42,
) -> "DSAv32Generator | GLM5Generator":
    """Load the matching backend .so and build the generator for ``model_type``.

    DeepSeek-V3.2 and GLM-5 ship as separate libraries; only one backend loads
    per process. Generators are imported lazily after the backend is loaded.
    """
    tilert.load_backend(model_type)

    if model_type == "deepseek_v3_2":
        from tilert.models.deepseek_v3_2.generator import DSAv32Generator
        from tilert.models.deepseek_v3_2.model_args import ModelArgs as DSAv32ModelArgs

        return DSAv32Generator(
            model_args=DSAv32ModelArgs(),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            model_weights_dir=model_weights_dir,
            with_mtp=with_mtp,
            top_p=top_p,
            top_k=top_k,
            use_topp=top_p < 1.0,
            sampling_seed=sampling_seed,
            enable_thinking=enable_thinking,
        )

    if model_type == "glm5":
        from tilert.models.glm_5.generator import GLM5Generator
        from tilert.models.glm_5.model_args import ModelArgsGLM5

        return GLM5Generator(
            model_args=ModelArgsGLM5(),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            model_weights_dir=model_weights_dir,
            with_mtp=with_mtp,
            top_p=top_p,
            top_k=top_k,
            use_topp=top_p < 1.0,
            enable_thinking=enable_thinking,
            sampling_seed=sampling_seed,
        )

    raise ValueError(f"unsupported model_type: {model_type!r}")


def parse_args():  # type: ignore
    parser = ArgumentParser(description="Command-line interface for text generation.")
    parser.add_argument(
        "--model-weights-dir",
        type=str,
        default=None,
        help="Path to model weights directory (resolved from ~/.tilert/config.toml if omitted)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek_v3_2",
        choices=["deepseek_v3_2", "glm5"],
        help="Model type to use (default: deepseek_v3_2).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4000, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling threshold. Use < 1.0 to enable top-p sampling (e.g. 0.9)",
    )
    parser.add_argument("--top-k", type=int, default=256, help="Top-k sampling threshold")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--with-mtp",
        action="store_true",
        help="Enable MTP (Multi-Token Prediction) for speculative decoding",
    )
    parser.add_argument(
        "--use-random-weights",
        action="store_true",
        help="Use random weights instead of pretrained (for testing MTP without real weights)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode in chat template",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=42,
        help="Sampling seed for top-p sampling (fixed per request, default: 42)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override display name for benchmark tables",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag for regression_plots/ directory (default: auto-detect from git state)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help="Comma-separated mode filters: top-k1,top-p0.95 (default: all)",
    )
    parser.add_argument(
        "--workloads",
        type=str,
        default=None,
        help="Comma-separated workload filters: short,coding,long (default: all)",
    )
    parser.add_argument(
        "--enable-logprobs",
        action="store_true",
        help="Enable kernel-level top-256 logprobs export (for benchmarking overhead)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """
    Usage (run as a module; --model-weights-dir may be omitted if the path is
    registered under ~/.tilert/config.toml). Run DeepSeek-V3.2 and GLM-5 in
    separate processes — the two backends cannot coexist in one interpreter.

    # DeepSeek-V3.2 — standard generation with pretrained weights:
    python -m tilert.generate --model deepseek_v3_2 \
        --model-weights-dir /path/to/DeepSeek-V3.2-TileRT \
        --max-new-tokens 1000 2>&1 | tee test.log

    # DeepSeek-V3.2 — MTP generation with random weights (for testing):
    python -m tilert.generate --model deepseek_v3_2 --with-mtp --use-random-weights \
        --model-weights-dir /path/to/DeepSeek-V3.2-TileRT \
        --max-new-tokens 1000 2>&1 | tee test.log

    # DeepSeek-V3.2 — MTP generation with pretrained weights:
    python -m tilert.generate --model deepseek_v3_2 --with-mtp \
        --model-weights-dir /path/to/DeepSeek-V3.2-TileRT \
        --max-new-tokens 1000 2>&1 | tee test.log

    # GLM-5 — standard generation:
    python -m tilert.generate --model glm5 \
        --model-weights-dir /path/to/GLM-5-FP8-TileRT \
        --max-new-tokens 1000 2>&1 | tee test.log

    # GLM-5 — MTP generation:
    python -m tilert.generate --model glm5 --with-mtp \
        --model-weights-dir /path/to/GLM-5-FP8-TileRT \
        --max-new-tokens 1000 2>&1 | tee test.log
    """
    args = parse_args()

    config_key = args.model
    model_name = args.model.upper()
    if args.model_name:
        model_name = args.model_name
    model_weights_dir = get_weights_dir(config_key, cli_override=args.model_weights_dir)

    if args.interactive:
        with_mtp = args.with_mtp
    else:
        with_mtp = True

    generator = get_generator(
        model_type=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        model_weights_dir=model_weights_dir,
        with_mtp=with_mtp,
        top_p=args.top_p,
        top_k=args.top_k,
        enable_thinking=args.enable_thinking,
        sampling_seed=args.sampling_seed,
    )

    t0 = time.monotonic()
    if args.use_random_weights:
        print("Initializing random weights...")
        if hasattr(generator, "init"):
            generator.init()  # type: ignore[union-attr]
        generator.init_random_weights()
    else:
        print("Loading pretrained weights...")
        generator.from_pretrained()
    load_time = time.monotonic() - t0

    if args.enable_logprobs:
        if hasattr(generator.decode_layer, "set_logprobs_enabled"):
            generator.decode_layer.set_logprobs_enabled(True)  # type: ignore[union-attr]
            print("Logprobs export enabled (top-256)")
        else:
            print(f"Warning: logprobs not supported for {type(generator).__name__}")

    if args.interactive:
        print("Welcome to the TileRT interactive mode! Type '/exit' to exit.")
        while True:
            prompt = input(">>> ")
            if prompt == "/exit":
                break
            _ = generator.generate(prompt)  # type: ignore[has-type]
    else:

        bench_top_p = args.top_p if args.top_p < 1.0 else 0.95
        modes = [
            BenchMode(with_mtp=False, label="top-k1 w/o MTP"),
            BenchMode(with_mtp=True, label="top-k1 w/ MTP"),
            BenchMode(
                with_mtp=True,
                label=f"top-p{bench_top_p} w/ MTP",
                use_topp=True,
                top_p=bench_top_p,
                top_k=args.top_k,
                temperature=args.temperature,
            ),
        ]

        if args.modes:
            allowed = {m.strip() for m in args.modes.split(",")}
            modes = [m for m in modes if any(a in m.label for a in allowed)]
            if not modes:
                raise SystemExit(
                    f"Error: --modes '{args.modes}' matched no benchmark modes. "
                    f"Valid tokens: top-k1, top-p0.95"
                )

        t0 = time.monotonic()
        workload_runners = []
        allowed_workloads = (
            {w.strip() for w in args.workloads.split(",")}
            if args.workloads
            else {"short", "coding", "long"}
        )
        if "short" in allowed_workloads:
            workload_runners.append(short_bench.run)
        if "coding" in allowed_workloads:
            workload_runners.append(coding_bench.run)
        if "long" in allowed_workloads:
            workload_runners.append(long_bench.run)
        if not workload_runners:
            raise SystemExit(
                f"Error: --workloads '{args.workloads}' matched no workloads. "
                f"Valid values: short, coding, long"
            )

        all_bench_results = [
            runner(generator, modes) for runner in workload_runners  # type: ignore[arg-type]
        ]
        bench_time = time.monotonic() - t0
        all_bench_stats = [stats for stats, _ in all_bench_results]

        print_summary_table(
            merge_stats(all_bench_stats),
            model_name=model_name,
        )

        total = load_time + bench_time
        print(f"\n## {model_name} Timing")
        print()
        print("| Phase | Time |")
        print("|-------|------|")
        print(f"| Loading | {load_time:.1f}s |")
        print(f"| Benchmark | {bench_time:.1f}s |")
        print(f"| **Total** | **{total:.1f}s** |")

    print("Cleaning up...")
    generator.cleanup()
