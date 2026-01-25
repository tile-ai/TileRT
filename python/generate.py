"""Text generation script for TileRT."""

from argparse import ArgumentParser
from typing import cast

import numpy as np

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
    return parser.parse_args()


if __name__ == "__main__":
    """
    usage:
    execute below command under tilert root directory:

    # Standard generation with pretrained weights:
    python python/generate.py --model-weights-dir "xxxx" 2>&1 | tee test.log

    # MTP generation with random weights (for testing):
    python python/generate.py --model-weights-dir "xxxx" --with-mtp \
        --use-random-weights 2>&1 | tee test.log

    # MTP generation with pretrained weights (when available):
    python python/generate.py --model-weights-dir "xxxx" --with-mtp 2>&1 | tee test.log
    """
    args = parse_args()

    generator: ShowHandsGenerator = ShowHandsGenerator(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        model_weights_dir=args.model_weights_dir,
        with_mtp=args.with_mtp,
    )

    if args.use_random_weights:
        print("Initializing with random weights...")
        generator.init_random_weights()
    else:
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
        # This prompt is to test the model’s ability to follow instructions
        # (in terms of quantity, type, and length) while keeping it fun.
        print("==== Performance ====")
        prompt = "Tell me 10 jokes, keep them all under 100 words."
        print("Prompt:", prompt)
        all_times = []
        all_accepted = []
        for _iter in range(20):
            if _iter % 5 == 0:
                print(f"Executing iter {_iter}...")
            results, time_list, accepted_counts = cast(
                tuple[str, list[float], list[int]],
                generator.generate(prompt, False),  # type: ignore[has-type]
            )
            all_times.append(time_list)
            all_accepted.append(accepted_counts)

        if args.with_mtp:
            for token_num in range(100, 300, 100):
                times_to_token_num = []
                for time_list, accepted_list in zip(all_times, all_accepted):
                    if len(time_list) > 5 and len(accepted_list) > 5:
                        times = time_list[5:]
                        accepted = accepted_list[5:]
                        cumsum_tokens = np.cumsum(accepted)
                        cumsum_times = np.cumsum(times)
                        # Find index where we reach token_num tokens
                        idx = np.searchsorted(cumsum_tokens, token_num)
                        if idx < len(cumsum_times):
                            times_to_token_num.append(cumsum_times[idx])
                if times_to_token_num:
                    mean_total_time = np.mean(times_to_token_num)
                    mean_time = mean_total_time / token_num
                    speed = 1 / mean_time
                    out_str = (
                        f"**Perf@{token_num}: {speed:.3f} tokens/s & "
                        f"{(mean_time * 1000):.3f} ms**"
                    )
                    print(out_str)

            # Print accepted tokens statistics
            flat_accepted = [a for accepted_list in all_accepted for a in accepted_list]
            if flat_accepted:
                avg_accepted = sum(flat_accepted) / len(flat_accepted)
                min_accepted = min(flat_accepted)
                max_accepted = max(flat_accepted)
                print(
                    f"**Accepted length: mean={avg_accepted:.2f}, "
                    f"min={min_accepted}, max={max_accepted}**"
                )
        else:
            all_times_np = np.array(all_times)
            for token_num in range(100, 300, 100):
                mean_time = np.mean(all_times_np[..., 5:token_num])
                speed = 1 / mean_time
                out_str = (
                    f"**Perf@{token_num}: {speed:.3f} tokens/s & {(mean_time * 1000):.3f} ms**"
                )
                print(out_str)
        print(results)

        # This prompt is used to test long sequence generation
        prompt = "Hi, can you tell me a very long story, with roughly 3000 words?"
        print("Prompt:", prompt)
        print("Completion:")
        completion, _, _ = generator.generate(prompt)  # type: ignore[has-type]

    print("Cleaning up...")
    generator.cleanup()
