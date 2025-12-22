"""Text generation script for TileRT."""

from argparse import ArgumentParser

from tilert.models.deepseek_v3_2.dsa_show_hands import ShowHandsGenerator


def parse_args():  # type: ignore
    parser = ArgumentParser(description="Command-line interface for text generation.")
    parser.add_argument(
        "--model-weights-dir",
        type=str,
        required=True,
        help="Path to model weights directory",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4000, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--fp8", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    """
    usage:
    execute below command under tilert root directory:

    python python/generate.py --model-weights-dir "xxxx" 2>&1 | tee test.log
    """
    args = parse_args()

    generator: ShowHandsGenerator = ShowHandsGenerator(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        model_weights_dir=args.model_weights_dir,
        enable_fp8_ops=args.fp8,
    )

    # uncomment to use random weights
    # generator.init_random_weights()

    # use pretrained weights
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
        # This prompt is to test the model’s ability to follow instructions
        # (in terms of quantity, type, and length) while keeping it fun.
        prompt = "Tell me 10 jokes, keep them all under 100 words."

        print("Prompt:", prompt)
        print("Completion:")
        completion: str = generator.generate(prompt)  # type: ignore[has-type]

        # This prompt is used to test long sequence generation
        prompt = "Hi, can you tell me a very long story, with roughly 3000 words?"
        print("Prompt:", prompt)
        print("Completion:")
        completion = generator.generate(prompt)  # type: ignore[has-type]
