"""Text generation script for TileRT."""

import math
import time
from argparse import ArgumentParser

import torch
from transformers import AutoTokenizer

from tilert import logger, tilert_init
from tilert.models.deepseek_v3_2.model_args import ModelArgs as ModelArgsV3_2
from tilert.models.deepseek_v3_2.modules.dsa_show_hands import (
    ShowHandsDSALayer,
    TempVars,
)
from tilert.models.deepseek_v3_2.params import IntermediateMapper

__all__ = [
    "ShowHandsGenerator",
    "parse_args",
]


def stats_time(time_list: list[float], title: str) -> None:
    if len(time_list) > 0:
        avg_time = sum(time_list) / len(time_list)
        # cal std-dev
        std_dev = math.sqrt(sum((x - avg_time) ** 2 for x in time_list) / len(time_list))
        logger.info(title)
        logger.info(f"--Average time taken to generate token: {avg_time * 1000:.4f} ms")
        logger.info(f"--Standard deviation of time: {std_dev * 1000:.4f} ms")
        logger.info(f"--Effective tokens per second: {1 / avg_time:.4f}")


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

    return parser.parse_args()


class ShowHandsGenerator:
    def __init__(
        self,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        model_weights_dir: str = "",
    ):
        """Initialize the ShowHandsGenerator.

        Args:
            max_new_tokens: Maximum number of new tokens to generate. Defaults to 100.
            temperature: Temperature for sampling. Defaults to 1.0.
            model_weights_dir: Path of the model weights directory.
        """
        torch.set_num_threads(64)
        self.model_weights_dir = model_weights_dir

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.config = ModelArgsV3_2()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_weights_dir)
        self.eos_id = self.tokenizer.eos_token_id
        self.batch_size = 1  # fixed batch size to 1 for now

        self.default_device = torch.device("cuda:0")

        self.decode_layer = ShowHandsDSALayer(
            max_seq_len=self.config.max_seq_len,
            model_path=self.model_weights_dir,
        )

    def init(self) -> None:
        """Initialize the ShowHandsGenerator."""
        tilert_init()

    def init_random_weights(self) -> None:
        """Random initialize the weights."""
        self.decode_layer.init_random_weights()

    def from_pretrained(self) -> None:
        """Load the model weights from the given path."""
        self.decode_layer.from_pretrained(self.model_weights_dir)

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        """Main function to load the model and perform single sequence generation."""
        prompt_tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )

        max_seq_len = self.config.max_seq_len
        prompt_len = len(prompt_tokens)
        total_len = min(max_seq_len, self.max_new_tokens + prompt_len)

        tokens = torch.full(
            (self.batch_size, total_len),
            -1,
            dtype=torch.long,
            device=self.default_device,
        )
        tokens[0, :prompt_len] = torch.tensor(
            prompt_tokens, dtype=torch.long, device=self.default_device
        )
        prompt_mask = tokens != -1

        prev_pos = 0
        finished = torch.tensor(
            [False] * self.batch_size, dtype=torch.bool, device=self.default_device
        )

        time_list = []
        for cur_pos_val in range(1, total_len):
            start_time = time.time()
            multi_devices_results = self.decode_layer.forward(tokens[0, prev_pos])
            end_time = time.time()
            time_list.append(end_time - start_time)

            intermediates, *_ = multi_devices_results[0]
            intermediates_mapper = IntermediateMapper(list(intermediates[-TempVars.num_params() :]))
            next_token = intermediates_mapper.token_out[0]

            # replace the next token with the prompt token if the prompt mask is True
            next_token = torch.where(
                prompt_mask[0, cur_pos_val], tokens[0, cur_pos_val], next_token
            )
            tokens[0, cur_pos_val] = next_token
            finished |= torch.logical_and(~prompt_mask[0, cur_pos_val], next_token == self.eos_id)
            prev_pos = cur_pos_val
            if cur_pos_val > prompt_len:
                decoded_tokens = self.tokenizer.decode(
                    [next_token.item()], skip_special_tokens=True
                )
                print(decoded_tokens, end="", flush=True)

            if finished.all():
                break

        print("\n")
        logger.info(f"--Number of tokens generated: {len(time_list)}")

        # skip the first several samples to avoid the warmup effect
        stats_time(time_list[5:], "==== Performance ====")
        print("\n")

        # Reset sequence after generation, i.e. reset the cur_pos to 0 internally
        self.decode_layer.reset_sequence()

        completion_tokens = []
        for _, toks in enumerate(tokens.tolist()):
            toks = toks[prompt_len : prompt_len + self.max_new_tokens]
            if self.eos_id in toks:
                toks = toks[: toks.index(self.eos_id)]
            completion_tokens.append(toks)

        decoded_tokens = self.tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)

        return decoded_tokens[0] if decoded_tokens else ""


if __name__ == "__main__":
    """
    usage:
    execute below command under tilert root directory:

    python python/generate.py --model-weights-dir "xxxx" 2>&1 | tee test.log
    """
    args = parse_args()

    generator = ShowHandsGenerator(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        model_weights_dir=args.model_weights_dir,
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
            _ = generator.generate(prompt)
    else:
        # This prompt is to test the model’s ability to follow instructions
        # (in terms of quantity, type, and length) while keeping it fun.
        prompt = """Tell me three jokes:
1. a dad joke,
2. a programmer joke,
3. a joke that only makes sense if you've ever tried to train a large language model.
Keep them all under 15 words.
"""
        print("Prompt:", prompt)
        print("Completion:")
        completion = generator.generate(prompt)

        # This prompt is used to test long sequence generation
        prompt = "Hi, can you tell me a very long story, with roughly 3000 words?"
        print("Prompt:", prompt)
        print("Completion:")
        completion = generator.generate(prompt)
