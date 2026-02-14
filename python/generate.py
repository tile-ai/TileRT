"""Text generation script for TileRT."""

from argparse import ArgumentParser

from benchmark import BenchMode
from benchmark import coding_prompt as coding_bench
from benchmark import long_prompt as long_bench
from benchmark import merge_stats, print_summary_table
from benchmark import short_prompt as short_bench

from tilert.models.deepseek_v3_2.generator import DSAv32Generator
from tilert.models.deepseek_v3_2.model_args import ModelArgs as DSAv32ModelArgs
from tilert.models.glm_5.generator import GLM5Generator
from tilert.models.glm_5.model_args import ModelArgsGLM5


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
) -> DSAv32Generator | GLM5Generator:
    """Get the appropriate generator based on model type."""
    assert model_type in ["deepseek_v3_2", "glm5"]
    if model_type == "deepseek_v3_2":
        model_args = DSAv32ModelArgs()
        return DSAv32Generator(
            model_args=model_args,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            model_weights_dir=model_weights_dir,
            with_mtp=with_mtp,
            top_p=top_p,
            top_k=top_k,
            use_topp=top_p < 1.0,
            sampling_seed=sampling_seed,
        )
    model_args = ModelArgsGLM5()
    return GLM5Generator(
        model_args=model_args,
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


def parse_args():  # type: ignore
    parser = ArgumentParser(description="Command-line interface for text generation.")
    parser.add_argument(
        "--model-weights-dir",
        type=str,
        required=True,
        help="Path to model weights directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek_v3_2",
        choices=["deepseek_v3_2", "glm5"],
        help="Model type to use (default: deepseek_v3_2)",
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
    return parser.parse_args()


if __name__ == "__main__":
    """
    usage:
    execute below command under tilert root directory:

    # DeepSeek V3.2 - Standard generation with pretrained weights:
    python python/generate.py --model-weights-dir "xxxx" 2>&1 | tee test.log

    # DeepSeek V3.2 - MTP generation with random weights (for testing):
    python python/generate.py --model-weights-dir "xxxx" --with-mtp \
        --use-random-weights 2>&1 | tee test.log

    # DeepSeek V3.2 - MTP generation with pretrained weights (when available):
    python python/generate.py --model-weights-dir "xxxx" --with-mtp 2>&1 | tee test.log

    # GLM5 - Standard generation with random weights (for testing):
    python python/generate.py --model glm5 --model-weights-dir "xxxx" \
        --use-random-weights 2>&1 | tee test.log

    # GLM5 - Standard generation with pretrained weights:
    python python/generate.py --model glm5 --model-weights-dir "xxxx" 2>&1 | tee test.log

    # GLM5 - MTP generation with random weights (for testing):
    python python/generate.py --model glm5 --model-weights-dir "xxxx" --with-mtp \
        --use-random-weights 2>&1 | tee test.log

    # GLM5 - MTP generation with pretrained weights:
    python python/generate.py --model glm5 --model-weights-dir "xxxx" --with-mtp \
        2>&1 | tee test.log
    """
    args = parse_args()

    generator = get_generator(
        model_type=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        model_weights_dir=args.model_weights_dir,
        with_mtp=args.with_mtp if args.interactive else True,
        top_p=args.top_p,
        top_k=args.top_k,
        enable_thinking=args.enable_thinking,
        sampling_seed=args.sampling_seed,
    )

    print("Loading pretrained weights...")
    generator.from_pretrained()

    # simple memoryless interactive mode
    if args.interactive:
        print("Welcome to the TileRT interactive mode! Type '/exit' to exit.")
        while True:
            prompt = input(">>> ")
            if prompt == "/exit":
                break
            _ = generator.generate(prompt)  # type: ignore[has-type]
    else:

        # 3 modes: top-k1 w/o MTP, top-k1 w/ MTP, top-p0.95 w/ MTP
        modes = [
            BenchMode(with_mtp=False, label="top-k1 w/o MTP"),
            BenchMode(with_mtp=True, label="top-k1 w/ MTP"),
            BenchMode(
                with_mtp=True,
                label="top-p0.95 w/ MTP",
                use_topp=True,
                top_p=0.95,
                top_k=args.top_k,
                temperature=args.temperature,
            ),
        ]

        # Run all benchmarks and collect stats
        all_bench_stats = [
            short_bench.run(generator, modes),
            coding_bench.run(generator, modes),
            long_bench.run(generator, modes),
        ]

        # Print unified summary table
        print_summary_table(
            merge_stats(all_bench_stats),
            model_name=args.model.upper(),
        )

    print("Cleaning up...")
    generator.cleanup()
