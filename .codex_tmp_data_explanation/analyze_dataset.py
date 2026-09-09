from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(r"C:\Users\MAHATVA GOEL\Desktop\Gait Analysis")
OUT_DIR = ROOT / ".codex_tmp_data_explanation"
SUBJECT_RE = re.compile(r"^Sub\d+_[AH]$")
TRIAL_RE = re.compile(r"_(\d+)(?:_?[LR])?$", re.IGNORECASE)
LIMB_RE = re.compile(r"_?[LR]$", re.IGNORECASE)


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * p
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def parse_file(path: Path) -> dict:
    is_insole = bool(LIMB_RE.search(path.name))
    expected_columns = 4 if is_insole else 18
    line_count = 0
    blank_lines = 0
    valid_rows = 0
    malformed_rows = 0
    nonnumeric_rows = 0
    column_counts: Counter[int] = Counter()
    trigger_counts: Counter[float] = Counter()
    times: list[float] = []
    dts: list[float] = []
    zero_dts = 0
    negative_dts = 0
    large_positive_gaps = 0
    min_signed_dt = math.inf
    invalid_examples: list[dict] = []
    unexpected_trigger_examples: list[dict] = []
    min_values = [math.inf] * expected_columns
    max_values = [-math.inf] * expected_columns
    sum_values = [0.0] * expected_columns
    sumsq_values = [0.0] * expected_columns

    previous_time: float | None = None
    previous_trigger: float | None = None
    trigger_rising_edges = 0
    trigger_falling_edges = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        for raw_line in stream:
            line_count += 1
            stripped = raw_line.strip("\r\n")
            if not stripped.strip():
                blank_lines += 1
                continue
            tokens = [token.strip() for token in stripped.split("\t") if token.strip()]
            column_counts[len(tokens)] += 1
            try:
                values = [float(token) for token in tokens]
            except ValueError:
                nonnumeric_rows += 1
                malformed_rows += 1
                if len(invalid_examples) < 3:
                    invalid_examples.append({"line": line_count, "reason": "nonnumeric", "text": stripped[:240]})
                continue
            if len(values) != expected_columns:
                malformed_rows += 1
                if len(invalid_examples) < 3:
                    invalid_examples.append({"line": line_count, "reason": f"{len(values)} columns", "text": stripped[:240]})
                continue

            valid_rows += 1
            current_time = values[0]
            times.append(current_time)
            trigger_counts[values[1]] += 1
            if previous_trigger is not None:
                if previous_trigger == 0.0 and values[1] == 1.0:
                    trigger_rising_edges += 1
                elif previous_trigger == 1.0 and values[1] == 0.0:
                    trigger_falling_edges += 1
            previous_trigger = values[1]
            if values[1] not in (0.0, 1.0) and len(unexpected_trigger_examples) < 3:
                unexpected_trigger_examples.append({"line": line_count, "values": values})
            if previous_time is not None:
                dt = current_time - previous_time
                min_signed_dt = min(min_signed_dt, dt)
                if dt < 0:
                    negative_dts += 1
                elif dt == 0:
                    zero_dts += 1
                else:
                    dts.append(dt)
                    if dt > 0.05:
                        large_positive_gaps += 1
            previous_time = current_time
            for index, value in enumerate(values):
                min_values[index] = min(min_values[index], value)
                max_values[index] = max(max_values[index], value)
                sum_values[index] += value
                sumsq_values[index] += value * value

    means = []
    stddevs = []
    for index in range(expected_columns):
        if valid_rows:
            mean = sum_values[index] / valid_rows
            variance = max(0.0, sumsq_values[index] / valid_rows - mean * mean)
            means.append(mean)
            stddevs.append(math.sqrt(variance))
        else:
            means.append(None)
            stddevs.append(None)

    trial_match = TRIAL_RE.search(path.name)
    trial_id = int(trial_match.group(1)) if trial_match else None
    side = None
    side_match = re.search(r"_?([LR])$", path.name, re.IGNORECASE)
    if side_match:
        side = side_match.group(1).upper()

    return {
        "subject": path.parent.name,
        "group": path.parent.name.rsplit("_", 1)[-1],
        "file": path.name,
        "relative_path": str(path.relative_to(ROOT)),
        "kind": "insole" if is_insole else "FMG",
        "side": side,
        "trial_id": trial_id,
        "bytes": path.stat().st_size,
        "line_count": line_count,
        "blank_lines": blank_lines,
        "valid_rows": valid_rows,
        "malformed_rows": malformed_rows,
        "nonnumeric_rows": nonnumeric_rows,
        "expected_columns": expected_columns,
        "column_count_distribution": dict(sorted(column_counts.items())),
        "first_time": times[0] if times else None,
        "last_time": times[-1] if times else None,
        "duration_seconds": (times[-1] - times[0]) if len(times) > 1 else None,
        "median_positive_dt": statistics.median(dts) if dts else None,
        "p05_positive_dt": quantile(dts, 0.05),
        "p95_positive_dt": quantile(dts, 0.95),
        "zero_time_steps": zero_dts,
        "negative_time_steps": negative_dts,
        "nonpositive_time_steps": zero_dts + negative_dts,
        "large_positive_gaps_over_0_05s": large_positive_gaps,
        "min_signed_dt": None if min_signed_dt == math.inf else min_signed_dt,
        "trigger_counts": {str(key): value for key, value in sorted(trigger_counts.items())},
        "nonzero_trigger_rows": sum(value for key, value in trigger_counts.items() if key != 0),
        "trigger_rising_edges": trigger_rising_edges,
        "trigger_falling_edges": trigger_falling_edges,
        "invalid_examples": invalid_examples,
        "unexpected_trigger_examples": unexpected_trigger_examples,
        "min": [None if value == math.inf else value for value in min_values],
        "max": [None if value == -math.inf else value for value in max_values],
        "mean": means,
        "stddev": stddevs,
    }


def summarize(files: list[dict]) -> dict:
    by_subject: dict[str, dict] = {}
    for subject in sorted({item["subject"] for item in files}):
        rows = [item for item in files if item["subject"] == subject]
        fmg = [item for item in rows if item["kind"] == "FMG"]
        insole = [item for item in rows if item["kind"] == "insole"]
        fmg_ids = sorted(item["trial_id"] for item in fmg if item["trial_id"] is not None)
        left_ids = sorted(item["trial_id"] for item in insole if item["side"] == "L" and item["trial_id"] is not None)
        right_ids = sorted(item["trial_id"] for item in insole if item["side"] == "R" and item["trial_id"] is not None)
        by_subject[subject] = {
            "group": rows[0]["group"],
            "files": len(rows),
            "fmg_files": len(fmg),
            "insole_files": len(insole),
            "fmg_trial_ids": fmg_ids,
            "left_trial_ids": left_ids,
            "right_trial_ids": right_ids,
            "missing_left_for_fmg": sorted(set(fmg_ids) - set(left_ids)),
            "missing_right_for_fmg": sorted(set(fmg_ids) - set(right_ids)),
            "extra_left_without_fmg": sorted(set(left_ids) - set(fmg_ids)),
            "extra_right_without_fmg": sorted(set(right_ids) - set(fmg_ids)),
            "valid_rows": sum(item["valid_rows"] for item in rows),
            "malformed_rows": sum(item["malformed_rows"] for item in rows),
            "bytes": sum(item["bytes"] for item in rows),
        }

    group_counts = Counter(item["group"] for item in files)
    kind_counts = Counter(item["kind"] for item in files)
    trigger_values: Counter[str] = Counter()
    for item in files:
        trigger_values.update(item["trigger_counts"])

    fmg_ranges = []
    insole_ranges = []
    for item in files:
        if item["kind"] == "FMG" and item["valid_rows"]:
            for channel in range(2, 18):
                fmg_ranges.append((item["min"][channel], item["max"][channel]))
        elif item["kind"] == "insole" and item["valid_rows"]:
            insole_ranges.append((item["min"][2], item["max"][2], item["min"][3], item["max"][3]))

    medians_by_kind = {}
    for kind in ("FMG", "insole"):
        vals = [item["median_positive_dt"] for item in files if item["kind"] == kind and item["median_positive_dt"] is not None]
        medians_by_kind[kind] = {
            "median_dt": statistics.median(vals) if vals else None,
            "min_file_median_dt": min(vals) if vals else None,
            "max_file_median_dt": max(vals) if vals else None,
        }

    return {
        "subject_count": len(by_subject),
        "healthy_subject_count": sum(1 for item in by_subject.values() if item["group"] == "H"),
        "amputee_subject_count": sum(1 for item in by_subject.values() if item["group"] == "A"),
        "file_count": len(files),
        "group_file_counts": dict(group_counts),
        "kind_file_counts": dict(kind_counts),
        "total_bytes": sum(item["bytes"] for item in files),
        "total_lines": sum(item["line_count"] for item in files),
        "total_valid_rows": sum(item["valid_rows"] for item in files),
        "total_malformed_rows": sum(item["malformed_rows"] for item in files),
        "total_blank_lines": sum(item["blank_lines"] for item in files),
        "files_with_malformed_rows": sum(1 for item in files if item["malformed_rows"]),
        "files_with_nonnumeric_rows": sum(1 for item in files if item["nonnumeric_rows"]),
        "files_with_nonpositive_time_steps": sum(1 for item in files if item["nonpositive_time_steps"]),
        "total_zero_time_steps": sum(item["zero_time_steps"] for item in files),
        "total_negative_time_steps": sum(item["negative_time_steps"] for item in files),
        "total_large_positive_gaps_over_0_05s": sum(item["large_positive_gaps_over_0_05s"] for item in files),
        "trigger_value_counts": dict(trigger_values),
        "total_trigger_rising_edges": sum(item["trigger_rising_edges"] for item in files),
        "total_trigger_falling_edges": sum(item["trigger_falling_edges"] for item in files),
        "sampling": medians_by_kind,
        "global_fmg_channel_min": min(row[0] for row in fmg_ranges) if fmg_ranges else None,
        "global_fmg_channel_max": max(row[1] for row in fmg_ranges) if fmg_ranges else None,
        "global_insole_cop_min": min(row[0] for row in insole_ranges) if insole_ranges else None,
        "global_insole_cop_max": max(row[1] for row in insole_ranges) if insole_ranges else None,
        "global_insole_vgrf_min": min(row[2] for row in insole_ranges) if insole_ranges else None,
        "global_insole_vgrf_max": max(row[3] for row in insole_ranges) if insole_ranges else None,
        "subjects": by_subject,
    }


def write_outputs(files: list[dict], summary: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "dataset_analysis.json").open("w", encoding="utf-8") as stream:
        json.dump({"summary": summary, "files": files}, stream, indent=2)

    fields = [
        "subject", "group", "file", "kind", "side", "trial_id", "bytes",
        "line_count", "valid_rows", "malformed_rows", "blank_lines",
        "expected_columns", "first_time", "last_time", "duration_seconds",
        "median_positive_dt", "nonpositive_time_steps", "nonzero_trigger_rows",
    ]
    with (OUT_DIR / "file_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(files)


def main() -> None:
    subject_dirs = sorted(path for path in ROOT.iterdir() if path.is_dir() and SUBJECT_RE.match(path.name))
    files = [parse_file(path) for subject in subject_dirs for path in sorted(subject.iterdir()) if path.is_file()]
    summary = summarize(files)
    write_outputs(files, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
