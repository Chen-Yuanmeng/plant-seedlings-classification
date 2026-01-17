from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def detect_project_root(start: Path) -> Path:
    markers = {"problem.md", "requirements.txt", "runall.sh", ".git"}
    current = start
    while True:
        if any((current / marker).exists() for marker in markers):
            return current
        if current.parent == current:
            return start
        current = current.parent


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = detect_project_root(SCRIPT_DIR)
LOG_ROOT = PROJECT_ROOT / "reports" / "tuning_logs"
FINE_SUBDIR = "fine"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "tuning_results"
PER_LOG_DIR = OUTPUT_DIR / "per_log"
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"
EPOCH_PATTERN = re.compile(
    rf"Epoch\s+(?P<epoch>\d+)/(?P<total>\d+)\s+\|\s+Train\s+(?P<train_loss>{FLOAT_PATTERN})/(?P<train_acc>{FLOAT_PATTERN})\s+\|\s+Val\s+(?P<val_loss>{FLOAT_PATTERN})/(?P<val_acc>{FLOAT_PATTERN})",
    re.IGNORECASE,
)


def unsanitize_value(token: str) -> str:
    return token.replace("m", "-").replace("p", ".")


def coerce_value(text: str):
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def parse_params(path: Path) -> Tuple[str, Dict[str, object]]:
    parts = path.stem.split("__")
    model = parts[0] if parts else "unknown"
    params: Dict[str, object] = {}
    for chunk in parts[1:]:
        if "_" not in chunk:
            continue
        key, raw_value = chunk.split("_", 1)
        value = coerce_value(unsanitize_value(raw_value))
        params[key] = value
    return model, params


def parse_epoch_entries(path: Path) -> List[Dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    entries: List[Dict[str, object]] = []
    for match in EPOCH_PATTERN.finditer(text):
        entries.append(
            {
                "epoch": int(match.group("epoch")),
                "total_epochs": int(match.group("total")),
                "train_loss": float(match.group("train_loss")),
                "train_acc": float(match.group("train_acc")),
                "val_loss": float(match.group("val_loss")),
                "val_acc": float(match.group("val_acc")),
            }
        )
    return entries


def write_epoch_csv(log_path: Path, entries: List[Dict[str, object]]) -> Path:
    relative = log_path.relative_to(LOG_ROOT)
    csv_path = (PER_LOG_DIR / relative).with_suffix(".csv")
    ensure_parent(csv_path)
    header = ["epoch", "total_epochs", "train_loss", "train_acc", "val_loss", "val_acc"]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)
    return csv_path


def collect_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not LOG_ROOT.exists():
        return rows
    model_dirs = [path for path in LOG_ROOT.iterdir() if path.is_dir()]
    for model_dir in sorted(model_dirs):
        fine_dir = model_dir / FINE_SUBDIR
        if not fine_dir.exists():
            continue
        for log_path in sorted(fine_dir.rglob("*.log")):
            model, params = parse_params(log_path)
            entries = parse_epoch_entries(log_path)
            if not entries:
                print(f"Skipping {log_path}: no epoch summaries found.")
                continue
            epoch_csv_path = write_epoch_csv(log_path, entries)
            best_entry = max(entries, key=lambda item: (item["val_acc"], -item["val_loss"]))
            relative_path = log_path.relative_to(PROJECT_ROOT)
            epoch_csv_rel = epoch_csv_path.relative_to(PROJECT_ROOT)
            row = {
                "model": model,
                "log_path": str(relative_path).replace("\\", "/"),
                "epoch_csv": str(epoch_csv_rel).replace("\\", "/"),
                "params": params,
                "best_epoch": best_entry["epoch"],
                "total_epochs": best_entry["total_epochs"],
                "best_val_loss": best_entry["val_loss"],
                "best_val_acc": best_entry["val_acc"],
            }
            rows.append(row)
    rows.sort(key=lambda item: (item["model"], -item["best_val_acc"]))
    return rows


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_model_csv(model: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    param_keys = sorted({key for row in rows for key in row["params"].keys()})
    base_columns = [
        "model",
        "log_path",
        "epoch_csv",
        "best_epoch",
        "total_epochs",
        "best_val_loss",
        "best_val_acc",
    ]
    header = base_columns + param_keys
    output_path = OUTPUT_DIR / f"{model}.csv"
    ensure_parent(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        for row in rows:
            csv_row = {column: row.get(column, "") for column in base_columns}
            for key in param_keys:
                csv_row[key] = row["params"].get(key, "")
            writer.writerow(csv_row)
    rel_path = output_path.relative_to(PROJECT_ROOT)
    print(f"Summary written to {rel_path}")


def main() -> None:
    rows = collect_rows()
    if not rows:
        print("No tuning logs were found; nothing to write.")
        return
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)
    for model, model_rows in grouped.items():
        write_model_csv(model, model_rows)


if __name__ == "__main__":
    main()
