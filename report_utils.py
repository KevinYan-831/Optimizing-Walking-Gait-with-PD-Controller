import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def write_json(path, payload):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, indent=2)


def write_csv(path, header, rows):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def create_report_dir(base_dir, root="reports"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(base_dir) / root / f"lab_report_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=False)
    return report_dir


def export_lab_report(
    report_dir,
    *,
    degrees,
    metrics,
    arrays,
    sample_case,
    feature_names=None,
):
    if feature_names is None:
        feature_names = ["rot", "lif", "dur", "kp", "kd"]

    report_dir = Path(report_dir)

    summary_payload = {
        "run_timestamp": datetime.now().isoformat(),
        "degrees": degrees,
        "metrics": metrics,
        "sample_case": sample_case,
    }
    write_json(report_dir / "summary_metrics.json", summary_payload)

    comparison_rows = [
        ["forward_vs_test", "distance", metrics["forward_vs_test_distance"]["mse"], metrics["forward_vs_test_distance"]["mae"], metrics["forward_vs_test_distance"]["r2"]],
        ["forward_vs_test", "heading", metrics["forward_vs_test_heading"]["mse"], metrics["forward_vs_test_heading"]["mae"], metrics["forward_vs_test_heading"]["r2"]],
        ["reverse_vs_test", "parameters_avg", metrics["reverse_vs_test_parameters"]["mse"], metrics["reverse_vs_test_parameters"]["mae"], metrics["reverse_vs_test_parameters"]["r2"]],
        ["original_vs_reverse", "distance_output", metrics["original_vs_reverse_distance"]["mse"], metrics["original_vs_reverse_distance"]["mae"], metrics["original_vs_reverse_distance"]["r2"]],
        ["original_vs_reverse", "heading_output", metrics["original_vs_reverse_heading"]["mse"], metrics["original_vs_reverse_heading"]["mae"], metrics["original_vs_reverse_heading"]["r2"]],
    ]
    write_csv(
        report_dir / "comparison_metrics.csv",
        ["comparison", "target", "mse", "mae", "r2"],
        comparison_rows,
    )

    n_test = len(arrays["y_distance_test"])
    forward_rows = []
    for i in range(n_test):
        forward_rows.append(
            [
                i,
                arrays["y_distance_test"][i],
                arrays["forward_pred_distance_test"][i],
                arrays["forward_pred_distance_test"][i] - arrays["y_distance_test"][i],
                arrays["y_heading_test"][i],
                arrays["forward_pred_heading_test"][i],
                arrays["forward_pred_heading_test"][i] - arrays["y_heading_test"][i],
            ]
        )
    write_csv(
        report_dir / "forward_vs_test.csv",
        [
            "sample_idx",
            "gt_distance",
            "pred_distance_forward",
            "error_distance",
            "gt_heading",
            "pred_heading_forward",
            "error_heading",
        ],
        forward_rows,
    )

    reverse_rows = []
    for i in range(n_test):
        row = [i]
        for j, name in enumerate(feature_names):
            gt_val = arrays["M_test"][i, j]
            pred_val = arrays["reverse_pred_params_test_clipped"][i, j]
            row.extend([gt_val, pred_val, pred_val - gt_val])
        reverse_rows.append(row)

    reverse_header = ["sample_idx"]
    for name in feature_names:
        reverse_header.extend([f"gt_{name}", f"pred_{name}", f"error_{name}"])
    write_csv(report_dir / "reverse_vs_test_params.csv", reverse_header, reverse_rows)

    ovsr_rows = []
    for i in range(n_test):
        ovsr_rows.append(
            [
                i,
                arrays["y_distance_test"][i],
                arrays["forward_pred_distance_test"][i],
                arrays["reverse_forward_distance_test"][i],
                arrays["reverse_forward_distance_test"][i] - arrays["forward_pred_distance_test"][i],
                arrays["y_heading_test"][i],
                arrays["forward_pred_heading_test"][i],
                arrays["reverse_forward_heading_test"][i],
                arrays["reverse_forward_heading_test"][i] - arrays["forward_pred_heading_test"][i],
            ]
        )
    write_csv(
        report_dir / "original_vs_reverse_outputs.csv",
        [
            "sample_idx",
            "gt_distance",
            "original_forward_distance",
            "reverse_to_forward_distance",
            "delta_reverse_minus_original_distance",
            "gt_heading",
            "original_forward_heading",
            "reverse_to_forward_heading",
            "delta_reverse_minus_original_heading",
        ],
        ovsr_rows,
    )

    write_json(
        report_dir / "raw_arrays.json",
        {
            "y_distance_test": arrays["y_distance_test"],
            "y_heading_test": arrays["y_heading_test"],
            "forward_pred_distance_test": arrays["forward_pred_distance_test"],
            "forward_pred_heading_test": arrays["forward_pred_heading_test"],
            "M_test": arrays["M_test"],
            "reverse_pred_params_test_clipped": arrays["reverse_pred_params_test_clipped"],
            "reverse_forward_distance_test": arrays["reverse_forward_distance_test"],
            "reverse_forward_heading_test": arrays["reverse_forward_heading_test"],
        },
    )
