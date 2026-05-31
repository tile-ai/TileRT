"""Benchmark suite for TileRT generation."""

from dataclasses import dataclass
from typing import TypeAlias

from tilert.models.deepseek_v3_2.generator import DSAv32Generator

Generator: TypeAlias = DSAv32Generator


@dataclass
class BenchMode:
    """Configuration for a single benchmark mode."""

    with_mtp: bool
    label: str
    use_topp: bool = False
    top_p: float = 1.0
    top_k: int = 256
    temperature: float = 1.0


@dataclass
class CellStats:
    """Stats for a single table cell (one mode x one benchmark column)."""

    tok_s: float = 0.0
    iters_s: str = "-"
    acc_rate: str = "-"


BenchStats = dict[str, dict[str, CellStats]]


@dataclass
class PerStepData:
    """Per-step timing data from a single generation run."""

    prompt_len: int
    time_list: list[float]
    accepted_counts: list[int]


PerStepDict = dict[str, dict[str, list[PerStepData]]]


def apply_mode(generator: Generator, mode: BenchMode) -> None:
    """Apply sampling parameters for a benchmark mode."""
    generator.update_sampling_params(
        temperature=mode.temperature,
        top_p=mode.top_p,
        top_k=mode.top_k,
        use_topp=mode.use_topp,
    )


def merge_stats(stats_list: list[BenchStats]) -> BenchStats:
    """Merge multiple benchmark stats dicts by mode label."""
    merged: BenchStats = {}
    for stats in stats_list:
        for mode, cols in stats.items():
            merged.setdefault(mode, {}).update(cols)
    return merged


def _fmt(number: float, suffix: str) -> str:
    return f"{number:.3f} {suffix}"


def print_summary_table(
    all_stats: BenchStats,
    model_name: str,
) -> None:
    """Print a markdown summary table from merged benchmark stats.

    Each mode occupies 3 rows: tok/s, ms, acc_rate.
    """
    if not all_stats:
        return

    col_keys: list[str] = []
    for cols in all_stats.values():
        for k in cols:
            if k not in col_keys:
                col_keys.append(k)

    ROW_LABELS = ["tok/s", "it/s", "acc"]

    formatted: dict[str, dict[str, list[str]]] = {}
    for mode, cols in all_stats.items():
        formatted[mode] = {}
        for k in col_keys:
            cell = cols.get(k)
            if cell is None:
                formatted[mode][k] = ["-", "-", "-"]
            else:
                formatted[mode][k] = [
                    _fmt(cell.tok_s, "tok/s"),
                    cell.iters_s,
                    cell.acc_rate,
                ]

    col_widths: dict[str, int] = {}
    for k in col_keys:
        w = len(k)
        for mode_cells in formatted.values():
            for row_str in mode_cells.get(k, ["-"]):
                w = max(w, len(row_str))
        col_widths[k] = w

    mode_width = max(len("Mode"), max(len(m) for m in all_stats))
    mode_width = max(mode_width, max(len(r) for r in ROW_LABELS))

    print(f"\n## Benchmark Summary ({model_name})\n")

    hdr = [f" {'Mode':<{mode_width}} "]
    hdr += [f" {k:<{col_widths[k]}} " for k in col_keys]
    print("|" + "|".join(hdr) + "|")

    sep = ["-" * (mode_width + 2)]
    sep += ["-" * (col_widths[k] + 2) for k in col_keys]
    print("|" + "|".join(sep) + "|")

    mode_list = list(all_stats.keys())
    for _, mode in enumerate(mode_list):
        for row_idx, _row_label in enumerate(ROW_LABELS):
            label = mode if row_idx == 0 else ""
            cells = [f" {label:<{mode_width}} "]
            for k in col_keys:
                cell_text = formatted[mode][k][row_idx]
                cells.append(f" {cell_text:<{col_widths[k]}} ")
            print("|" + "|".join(cells) + "|")
