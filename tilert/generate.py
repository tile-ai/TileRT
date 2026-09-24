"""TileRT offline generation CLI."""

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import torch

import tilert

_MODEL_PACKAGES: dict[str, str] = {
    "deepseek_v3_2": "tilert.models.deepseek_v3_2",
    "glm5": "tilert.models.glm_5",
    "glm5_2_rocm": "tilert.models.glm_5_2_rocm",
}


def _release_products() -> set[str] | None:
    try:
        op = torch.ops.tilert.supported_models.default
        keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)
        return set(op.redispatch(keyset))
    except Exception:
        return None


def available_models() -> list[str]:
    present = [
        model
        for model, package in _MODEL_PACKAGES.items()
        if importlib.util.find_spec(package) is not None
    ]
    products = _release_products()
    if products is None:
        return present
    filtered = [m for m in present if _MODEL_PACKAGES[m].rsplit(".", 1)[-1] in products]
    return filtered or present


def get_generator(model: str, weights_dir: str, args: argparse.Namespace) -> Any:
    tilert.load_backend(model)
    use_topp = args.top_p < 1.0
    if model == "glm5_2_rocm":
        from tilert.models.glm_5_2_rocm.generator import Glm52Generator
        from tilert.models.glm_5_2_rocm.model_args import ModelArgsGlm52

        return Glm52Generator(
            model_args=ModelArgsGlm52(),
            model_weights_dir=weights_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_topp=use_topp,
            sampling_seed=args.sampling_seed,
            num_mtp=args.num_mtp,
            max_seq_len=args.max_seq_len,
        )
    if model == "glm5":
        from tilert.models.glm_5.generator import GLM5Generator
        from tilert.models.glm_5.model_args import ModelArgsGLM5

        return GLM5Generator(
            model_args=ModelArgsGLM5(),
            model_weights_dir=weights_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            use_topp=use_topp,
            sampling_seed=args.sampling_seed,
            with_mtp=args.num_mtp > 0,
        )
    if model == "deepseek_v3_2":
        from tilert.models.deepseek_v3_2.generator import DSAv32Generator
        from tilert.models.deepseek_v3_2.model_args import ModelArgs

        return DSAv32Generator(
            model_args=ModelArgs(),
            model_weights_dir=weights_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            use_topp=use_topp,
            sampling_seed=args.sampling_seed,
            enable_thinking=args.enable_thinking,
            with_mtp=args.num_mtp > 0,
        )
    raise SystemExit(
        f"[generate] model {model!r} is not offered by this build; available: {available_models()}"
    )


def _prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        text = Path(args.prompt_file).read_text(encoding="utf-8")
        out = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not out:
            raise SystemExit(f"[generate] no prompts in {args.prompt_file}")
        return out
    return [args.prompt]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    models = available_models()
    if not models:
        raise SystemExit(
            "[generate] this install ships no model package; the engine library may have failed to load (see the warning from `import tilert`)"
        )
    p = argparse.ArgumentParser(
        prog="python -m tilert.generate", description=__doc__.splitlines()[0]
    )
    p.add_argument("--model", choices=models, default=models[0])
    p.add_argument("--model-weights-dir", required=True, help="converted weights dir")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--prompt", default="Hello! Tell me about yourself.")
    src.add_argument(
        "--prompt-file", help="file of prompts, separated by blank lines; each is generated in turn"
    )
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help="KV cache length; honoured by the ROCm models (the CUDA ones take it from their model args)",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95, help="1.0 selects greedy (argmax) decoding")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--sampling-seed", type=int, default=42)
    p.add_argument(
        "--num-mtp", type=int, default=0, help="MTP draft depth; 0 disables speculative decoding"
    )
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--quiet", action="store_true", help="only print completions")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompts = _prompts(args)
    t0 = time.monotonic()
    generator = get_generator(args.model, args.model_weights_dir, args)
    generator.init()
    generator.from_pretrained()
    if not args.quiet:
        print(f"[generate] {args.model} loaded in {time.monotonic() - t0:.1f}s")
    try:
        for i, prompt in enumerate(prompts):
            if not args.quiet:
                print(f"\n=== prompt {i + 1}/{len(prompts)} ===\n{prompt}\n--- output ---")
            t = time.monotonic()
            text, times, accepts, prompt_len = generator.generate(prompt, print_log=False)
            wall = time.monotonic() - t
            print(text)
            if not args.quiet:
                n_tok = sum(accepts) if accepts else len(times)
                rate = n_tok / wall if wall > 0 else 0.0
                line = f"[generate] {prompt_len} prompt tok -> {n_tok} tok in {wall:.1f}s ({rate:.1f} tok/s)"
                if accepts:
                    line += f", mean accept {sum(accepts) / len(accepts):.2f}"
                print(line)
    finally:
        generator.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
