"""Long-prompt benchmark: single generation, measures long-form throughput."""

from typing import cast

import numpy as np
from benchmark import BenchMode, BenchStats, CellStats, Generator, apply_mode

PROMPT = "Hi, can you tell me a very long story, with roughly 3000 words?"


def run(generator: Generator, modes: list[BenchMode]) -> BenchStats:
    """Run the long-prompt benchmark for each mode.

    Returns stats with column: Long.
    """
    stats: BenchStats = {}

    for mode in modes:
        apply_mode(generator, mode)
        print(f"\n--- Long-prompt benchmark ({mode.label}) ---")
        print(f"Prompt: {PROMPT}")
        print("Completion:")

        _, time_list, accepted_counts = cast(
            tuple[str, list[float], list[int]],
            generator.generate(PROMPT, True, with_mtp=mode.with_mtp),
        )

        mode_stats: dict[str, CellStats] = {}

        if mode.with_mtp and accepted_counts:
            total_tokens = sum(accepted_counts)
            total_time = sum(time_list)
            speed = total_tokens / total_time if total_time > 0 else 0
            avg_ms = total_time / len(time_list) * 1000
            avg_a = total_tokens / len(accepted_counts)
            acc_rate = f"{avg_a:.2f}/{min(accepted_counts)}/{max(accepted_counts)}"
            mode_stats["Long"] = CellStats(tok_s=speed, ms=avg_ms, acc_rate=acc_rate)
        elif time_list:
            mean_time = float(np.mean(time_list))
            speed = 1 / mean_time
            mode_stats["Long"] = CellStats(tok_s=speed, ms=mean_time * 1000)

        stats[mode.label] = mode_stats

    return stats
