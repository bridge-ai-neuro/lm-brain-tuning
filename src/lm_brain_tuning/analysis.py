from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import atomic_write_json

PAPER_SYSTEM_ORDER = ("joint", "brain", "stimulus", "pretrained")


def _read_scores(paths: list[str | Path]) -> list[dict[str, str]]:
    required = {
        "task",
        "subfield",
        "system",
        "comparison_unit",
        "analysis_unit",
        "aggregation_unit",
        "model_name",
        "seed",
        "score",
    }
    rows: list[dict[str, str]] = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Normalized Holmes CSV must contain {sorted(required)}: {path}")
            for row_number, row in enumerate(reader, start=2):
                identity_fields = (
                    "task",
                    "subfield",
                    "system",
                    "comparison_unit",
                    "analysis_unit",
                    "aggregation_unit",
                    "model_name",
                )
                if any(not row.get(field) for field in identity_fields):
                    raise ValueError(f"Missing Holmes identity at {path}:{row_number}")
                try:
                    int(row["seed"])
                    score = float(row["score"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid Holmes value at {path}:{row_number}") from error
                if not math.isfinite(score):
                    raise ValueError(f"Non-finite Holmes score at {path}:{row_number}")
                rows.append(row)
    if not rows:
        raise ValueError("At least one Holmes score row is required")
    identities = [
        (
            row["comparison_unit"],
            row["aggregation_unit"],
            row["system"],
            row["task"],
            row["model_name"],
            int(row["seed"]),
        )
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate system/task/model/seed rows in Holmes inputs")
    return rows


def _validate_score_matrix(rows: list[dict[str, str]]) -> None:
    expected_seeds = {0, 1, 2, 3, 4}
    expected_systems = {row["system"] for row in rows}
    expected_tasks = {row["task"] for row in rows}
    systems_by_unit: dict[str, set[str]] = defaultdict(set)
    tasks_by_unit_system: dict[tuple[str, str], set[str]] = defaultdict(set)
    models_by_unit_system: dict[tuple[str, str], set[str]] = defaultdict(set)
    analysis_units_by_comparison: dict[str, set[str]] = defaultdict(set)
    aggregation_units_by_comparison: dict[str, set[str]] = defaultdict(set)
    aggregation_units_by_analysis: dict[str, set[str]] = defaultdict(set)
    subfields_by_task: dict[str, set[str]] = defaultdict(set)
    seeds_by_model_task: dict[tuple[str, str, str, str], set[int]] = defaultdict(set)
    for row in rows:
        comparison = row["comparison_unit"]
        system = row["system"]
        systems_by_unit[comparison].add(system)
        tasks_by_unit_system[(comparison, system)].add(row["task"])
        models_by_unit_system[(comparison, system)].add(row["model_name"])
        analysis_units_by_comparison[comparison].add(row["analysis_unit"])
        aggregation_units_by_comparison[comparison].add(row["aggregation_unit"])
        aggregation_units_by_analysis[row["analysis_unit"]].add(row["aggregation_unit"])
        subfields_by_task[row["task"]].add(row["subfield"])
        seeds_by_model_task[(comparison, system, row["task"], row["model_name"])].add(
            int(row["seed"])
        )
    incomplete_systems = {
        unit: sorted(expected_systems.difference(systems))
        for unit, systems in systems_by_unit.items()
        if systems != expected_systems
    }
    if incomplete_systems:
        raise ValueError(f"Every comparison unit requires every system: {incomplete_systems}")
    incomplete_tasks = {
        "/".join(identity): sorted(expected_tasks.difference(tasks))
        for identity, tasks in tasks_by_unit_system.items()
        if tasks != expected_tasks
    }
    if incomplete_tasks:
        raise ValueError(f"Every comparison unit requires every Holmes task: {incomplete_tasks}")
    ambiguous_models = {
        "/".join(identity): sorted(models)
        for identity, models in models_by_unit_system.items()
        if len(models) != 1
    }
    if ambiguous_models:
        raise ValueError(f"A comparison unit/system must identify one model: {ambiguous_models}")
    ambiguous_analysis_units = {
        comparison: sorted(units)
        for comparison, units in analysis_units_by_comparison.items()
        if len(units) != 1
    }
    if ambiguous_analysis_units:
        raise ValueError(
            "Each comparison unit must belong to one analysis unit: "
            f"{ambiguous_analysis_units}"
        )
    ambiguous_aggregation_units = {
        comparison: sorted(units)
        for comparison, units in aggregation_units_by_comparison.items()
        if len(units) != 1
    }
    if ambiguous_aggregation_units:
        raise ValueError(
            "Each comparison unit must belong to one aggregation unit: "
            f"{ambiguous_aggregation_units}"
        )
    aggregation_unit_counts = {
        analysis_unit: len(units)
        for analysis_unit, units in aggregation_units_by_analysis.items()
    }
    if len(set(aggregation_unit_counts.values())) != 1:
        raise ValueError(
            "Every analysis unit requires the same number of model-family units: "
            f"{aggregation_unit_counts}"
        )
    ambiguous_subfields = {
        task: sorted(subfields)
        for task, subfields in subfields_by_task.items()
        if len(subfields) != 1
    }
    if ambiguous_subfields:
        raise ValueError(f"Each Holmes task must belong to one subfield: {ambiguous_subfields}")
    incomplete_seeds = {
        "/".join(identity): sorted(expected_seeds.difference(seeds))
        for identity, seeds in seeds_by_model_task.items()
        if seeds != expected_seeds
    }
    if incomplete_seeds:
        raise ValueError(f"Every Holmes model/task requires seeds 0-4: {incomplete_seeds}")


def holm_correction(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, original_index in enumerate(order):
        running = max(running, (count - rank) * values[original_index])
        adjusted_sorted[rank] = min(running, 1.0)
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]


def analyze_holmes(
    inputs: list[str | Path],
    *,
    baseline: str,
    output_dir: str | Path,
) -> Path:
    from scipy.stats import ttest_ind, wilcoxon

    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty analysis output: {output_dir}")
    rows = _read_scores(inputs)
    _validate_score_matrix(rows)
    available_systems = {row["system"] for row in rows}
    systems = [system for system in PAPER_SYSTEM_ORDER if system in available_systems]
    systems.extend(sorted(available_systems.difference(PAPER_SYSTEM_ORDER)))
    if len(systems) < 2:
        raise ValueError("Holmes analysis requires at least two systems")
    if baseline not in systems:
        raise ValueError(f"Baseline {baseline!r} not found; available systems: {systems}")
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    analysis_unit_by_comparison: dict[str, str] = {}
    aggregation_unit_by_comparison: dict[str, str] = {}
    subfield_by_task: dict[str, str] = {}
    for row in rows:
        comparison = row["comparison_unit"]
        grouped[(comparison, row["system"], row["task"])].append(float(row["score"]))
        analysis_unit_by_comparison[comparison] = row["analysis_unit"]
        aggregation_unit_by_comparison[comparison] = row["aggregation_unit"]
        subfield_by_task[row["task"]] = row["subfield"]
    comparison_units = sorted(analysis_unit_by_comparison)
    tasks = sorted({task for _, _, task in grouped})

    comparisons: list[dict[str, object]] = []
    task_win_rates: dict[tuple[str, str, str], float] = {}
    for comparison_unit in comparison_units:
        for system in systems:
            for task in tasks:
                current = grouped[(comparison_unit, system, task)]
                wins = 0
                for reference_system in systems:
                    if reference_system == system:
                        continue
                    reference = grouped[(comparison_unit, reference_system, task)]
                    statistic, p_value = ttest_ind(
                        current, reference, equal_var=True, alternative="greater"
                    )
                    current_mean = float(np.mean(current))
                    reference_mean = float(np.mean(reference))
                    win = bool(p_value < 0.05 and current_mean > reference_mean)
                    wins += int(win)
                    comparisons.append(
                        {
                            "comparison_unit": comparison_unit,
                            "analysis_unit": analysis_unit_by_comparison[comparison_unit],
                            "system": system,
                            "reference": reference_system,
                            "task": task,
                            "mean": current_mean,
                            "reference_mean": reference_mean,
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                            "win": win,
                        }
                    )
                task_win_rates[(comparison_unit, system, task)] = wins / (len(systems) - 1)

    analysis_units = sorted(set(analysis_unit_by_comparison.values()))
    analysis_unit_win_rates: dict[tuple[str, str], float] = {}
    aggregation_unit_win_rates: dict[tuple[str, str, str], float] = {}
    for analysis_unit in analysis_units:
        aggregation_units = sorted(
            {
                aggregation_unit_by_comparison[comparison]
                for comparison in comparison_units
                if analysis_unit_by_comparison[comparison] == analysis_unit
            }
        )
        for aggregation_unit in aggregation_units:
            unit_comparisons = [
                comparison
                for comparison in comparison_units
                if analysis_unit_by_comparison[comparison] == analysis_unit
                and aggregation_unit_by_comparison[comparison] == aggregation_unit
            ]
            for system in systems:
                per_task = {
                    task: float(
                        np.mean(
                            [
                                task_win_rates[(comparison, system, task)]
                                for comparison in unit_comparisons
                            ]
                        )
                    )
                    for task in tasks
                }
                subfields = sorted(set(subfield_by_task.values()))
                per_subfield = [
                    float(
                        np.mean(
                            [
                                per_task[task]
                                for task in tasks
                                if subfield_by_task[task] == subfield
                            ]
                        )
                    )
                    for subfield in subfields
                ]
                aggregation_unit_win_rates[(analysis_unit, aggregation_unit, system)] = (
                    float(np.mean(per_subfield))
                )
        for system in systems:
            analysis_unit_win_rates[(analysis_unit, system)] = float(
                np.mean(
                    [
                        aggregation_unit_win_rates[(analysis_unit, aggregation_unit, system)]
                        for aggregation_unit in aggregation_units
                    ]
                )
            )
    win_rates = {
        system: float(
            np.mean(
                [
                    analysis_unit_win_rates[(analysis_unit, system)]
                    for analysis_unit in analysis_units
                ]
            )
        )
        for system in systems
    }

    pairwise: list[dict[str, object]] = []
    compared_systems = sorted(win_rates)
    raw_p: list[float] = []
    pending: list[tuple[str, str, float]] = []
    per_analysis_unit_scores = {
        system: [
            analysis_unit_win_rates[(analysis_unit, system)] for analysis_unit in analysis_units
        ]
        for system in compared_systems
    }
    for left_index, left in enumerate(compared_systems):
        for right in compared_systems[left_index + 1 :]:
            left_values = per_analysis_unit_scores[left]
            right_values = per_analysis_unit_scores[right]
            if np.allclose(left_values, right_values):
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = wilcoxon(left_values, right_values, alternative="two-sided")
            raw_p.append(float(p_value))
            pending.append((left, right, float(statistic)))
    corrected = holm_correction(raw_p) if raw_p else []
    for (left, right, statistic), p_value in zip(pending, corrected, strict=False):
        pairwise.append(
            {"left": left, "right": right, "statistic": statistic, "p_value_holm": p_value}
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "task_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "comparison_unit",
            "analysis_unit",
            "system",
            "reference",
            "task",
            "mean",
            "reference_mean",
            "statistic",
            "p_value",
            "win",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    with (output_dir / "task_win_rates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "comparison_unit",
                "analysis_unit",
                "system",
                "task",
                "win_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(
            {
                "comparison_unit": comparison_unit,
                "analysis_unit": analysis_unit_by_comparison[comparison_unit],
                "system": system,
                "task": task,
                "win_rate": task_win_rates[(comparison_unit, system, task)],
            }
            for comparison_unit in comparison_units
            for system in systems
            for task in tasks
        )
    summary = {
        "baseline": baseline,
        "win_rates": win_rates,
        "analysis_unit_win_rates": {
            analysis_unit: {
                system: analysis_unit_win_rates[(analysis_unit, system)] for system in systems
            }
            for analysis_unit in analysis_units
        },
        "aggregation_unit_win_rates": {
            analysis_unit: {
                aggregation_unit: {
                    system: aggregation_unit_win_rates[
                        (analysis_unit, aggregation_unit, system)
                    ]
                    for system in systems
                }
                for aggregation_unit in sorted(
                    {
                        unit
                        for subject, unit, _ in aggregation_unit_win_rates
                        if subject == analysis_unit
                    }
                )
            }
            for analysis_unit in analysis_units
        },
        "pairwise_wilcoxon": pairwise,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    _plot_win_rates(
        win_rates,
        analysis_unit_win_rates,
        analysis_units,
        output_dir / "win-rates.png",
    )
    return output_dir


def _plot_win_rates(
    win_rates: dict[str, float],
    analysis_unit_win_rates: dict[tuple[str, str], float],
    analysis_units: list[str],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(win_rates)
    values = [win_rates[label] for label in labels]
    errors = [
        float(
            np.std([analysis_unit_win_rates[(unit, label)] for unit in analysis_units])
            / np.sqrt(len(analysis_units))
        )
        for label in labels
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, values, yerr=errors, color="#4C78A8", capsize=5)
    axis.set_ylabel("Average win rate")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)
