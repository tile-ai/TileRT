"""Long-prompt benchmark: single generation, measures long-form throughput."""

from typing import cast

import numpy as np

from tilert.benchmark import (
    BenchMode,
    BenchStats,
    CellStats,
    Generator,
    PerStepData,
    PerStepDict,
    apply_mode,
)

PROMPT = "Hi, can you tell me a very long story, with roughly 3000 words?"


def run(generator: Generator, modes: list[BenchMode]) -> tuple[BenchStats, PerStepDict]:
    """Run the long-prompt benchmark for each mode.

    Returns stats with column: Long.
    """
    stats: BenchStats = {}
    per_step: PerStepDict = {}

    for mode in modes:
        apply_mode(generator, mode)
        print(f"\n--- Long-prompt benchmark ({mode.label}) ---")
        print(f"Prompt: {PROMPT}")
        print("Completion:")

        _, time_list, accepted_counts, prompt_len = cast(
            tuple[str, list[float], list[int], int],
            generator.generate(PROMPT, True, with_mtp=mode.with_mtp),
        )

        mode_stats: dict[str, CellStats] = {}

        if mode.with_mtp and accepted_counts:
            total_tokens = sum(accepted_counts)
            total_time = sum(time_list)
            speed = total_tokens / total_time if total_time > 0 else 0
            avg_a = total_tokens / len(accepted_counts)
            acc_rate = f"{avg_a:.2f}/{min(accepted_counts)}/{max(accepted_counts)}"

            cumtok = list(np.cumsum(accepted_counts))
            split_idx = next((i for i, t in enumerate(cumtok) if t >= 2048), len(time_list))
            end_idx = next((i for i, t in enumerate(cumtok) if t >= 2048 + 512), len(time_list))
            pre_time = time_list[:split_idx]
            post_time = time_list[split_idx:end_idx]
            pre_ips = len(pre_time) / sum(pre_time) if pre_time else 0.0
            post_ips = len(post_time) / sum(post_time) if post_time else 0.0
            iters_s = f"{pre_ips:.1f}/{post_ips:.1f} it/s"

            mode_stats["Long"] = CellStats(tok_s=speed, iters_s=iters_s, acc_rate=acc_rate)
        elif time_list:
            mean_time = float(np.mean(time_list))
            speed = 1 / mean_time

            split_idx = min(2048, len(time_list))
            end_idx = min(2048 + 512, len(time_list))
            pre_time = time_list[:split_idx]
            post_time = time_list[split_idx:end_idx]
            pre_ips = len(pre_time) / sum(pre_time) if pre_time else 0.0
            post_ips = len(post_time) / sum(post_time) if post_time else 0.0
            iters_s = f"{pre_ips:.1f}/{post_ips:.1f} it/s"

            mode_stats["Long"] = CellStats(tok_s=speed, iters_s=iters_s)

        per_step[mode.label] = {
            "Long": [
                PerStepData(
                    prompt_len=prompt_len, time_list=time_list, accepted_counts=accepted_counts
                )
            ]
        }

        stats[mode.label] = mode_stats

    return stats, per_step
