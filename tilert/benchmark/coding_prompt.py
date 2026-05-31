"""Coding-prompt benchmark: single generation, measures coding task throughput."""

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

PROMPT = "Hi, can you write a sort program in C for me?"


def run(generator: Generator, modes: list[BenchMode]) -> tuple[BenchStats, PerStepDict]:
    """Run the coding-prompt benchmark for each mode.

    Returns stats with column: Coding.
    """
    stats: BenchStats = {}
    per_step: PerStepDict = {}

    for mode in modes:
        apply_mode(generator, mode)
        print(f"\n--- Coding-prompt benchmark ({mode.label}) ---")
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
            iters_s = len(time_list) / total_time if total_time > 0 else 0.0
            mode_stats["Coding"] = CellStats(
                tok_s=speed, iters_s=f"{iters_s:.1f} it/s", acc_rate=acc_rate
            )
        elif time_list:
            mean_time = float(np.mean(time_list))
            speed = 1 / mean_time
            mode_stats["Coding"] = CellStats(tok_s=speed, iters_s=f"{speed:.1f} it/s")

        per_step[mode.label] = {
            "Coding": [
                PerStepData(
                    prompt_len=prompt_len, time_list=time_list, accepted_counts=accepted_counts
                )
            ]
        }

        stats[mode.label] = mode_stats

    return stats, per_step
