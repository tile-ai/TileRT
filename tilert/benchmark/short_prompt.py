"""Short-prompt benchmark: 1 warmup + 20 iterations, measures steady-state decode throughput."""

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

PROMPT = "Tell me 10 jokes, keep them all under 100 words."
NUM_ITERS = 20
TOKEN_CHECKPOINTS = [200]


def run(generator: Generator, modes: list[BenchMode]) -> tuple[BenchStats, PerStepDict]:
    """Run the short-prompt benchmark for each mode.

    Returns stats with columns: Short@<N> for each checkpoint.
    """
    stats: BenchStats = {}
    per_step: PerStepDict = {}

    for mode in modes:
        apply_mode(generator, mode)
        print(f"\n--- Short-prompt benchmark ({mode.label}) ---", flush=True)

        print("  warmup...", flush=True)
        generator.generate(PROMPT, False, with_mtp=mode.with_mtp)

        all_times: list[list[float]] = []
        all_accepted: list[list[int]] = []
        all_results: list[str] = []
        all_per_step_data: list[PerStepData] = []
        for _iter in range(NUM_ITERS):
            if _iter % 5 == 0:
                print(f"  iter {_iter}/{NUM_ITERS}...", flush=True)
            result, time_list, accepted_counts, prompt_len = cast(
                tuple[str, list[float], list[int], int],
                generator.generate(PROMPT, False, with_mtp=mode.with_mtp),
            )
            all_times.append(time_list)
            all_accepted.append(accepted_counts)
            all_results.append(result)
            all_per_step_data.append(
                PerStepData(
                    prompt_len=prompt_len, time_list=time_list, accepted_counts=accepted_counts
                )
            )

        mismatches = [i for i, r in enumerate(all_results) if r != all_results[0]]
        if mismatches:
            print(f"  WARNING: non-deterministic output at iters {mismatches}")
        print(f"Prompt: {PROMPT}")
        print(f"Completion:\n{all_results[0]}")

        mode_stats: dict[str, CellStats] = {}

        if mode.with_mtp:
            for token_num in TOKEN_CHECKPOINTS:
                speeds: list[float] = []
                iter_rates: list[float] = []
                for time_list, accepted_list in zip(all_times, all_accepted):
                    if time_list and accepted_list:
                        cumsum_tokens = np.cumsum(accepted_list)
                        cumsum_times = np.cumsum(time_list)
                        idx = int(np.searchsorted(cumsum_tokens, token_num))
                        if idx >= len(cumsum_times):
                            idx = len(cumsum_times) - 1
                        tok_count = int(cumsum_tokens[idx])
                        elapsed = float(cumsum_times[idx])
                        if elapsed > 0:
                            speeds.append(tok_count / elapsed)
                            iter_rates.append((idx + 1) / elapsed)
                if speeds:
                    speed = float(np.mean(speeds))

                    flat_accepted = [a for al in all_accepted for a in al]
                    acc_rate = "-"
                    if flat_accepted:
                        avg_a = sum(flat_accepted) / len(flat_accepted)
                        acc_rate = f"{avg_a:.2f}/{min(flat_accepted)}/{max(flat_accepted)}"

                    iters_s = float(np.mean(iter_rates)) if iter_rates else 0.0
                    mode_stats[f"Short@{token_num}"] = CellStats(
                        tok_s=speed,
                        iters_s=f"{iters_s:.1f} it/s",
                        acc_rate=acc_rate,
                    )
        else:
            for token_num in TOKEN_CHECKPOINTS:
                per_token_times = []
                for time_list in all_times:
                    trimmed = time_list[:token_num]
                    if trimmed:
                        per_token_times.extend(trimmed)
                if per_token_times:
                    mean_time = float(np.mean(per_token_times))
                    speed = 1 / mean_time
                    mode_stats[f"Short@{token_num}"] = CellStats(
                        tok_s=speed, iters_s=f"{speed:.1f} it/s"
                    )

        mode_per_step: dict[str, list[PerStepData]] = {}
        for token_num in TOKEN_CHECKPOINTS:
            mode_per_step[f"Short@{token_num}"] = all_per_step_data
        per_step[mode.label] = mode_per_step

        stats[mode.label] = mode_stats

    return stats, per_step
