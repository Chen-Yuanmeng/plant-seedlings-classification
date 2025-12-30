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
OUTPUT_DIR = PROJECT_ROOT / "reports" / "tuning_results"
EPOCH_PATTERN = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\|\s+Train\s+([0-9.]+)/([0-9.]+)\s+\|\s+Val\s+([0-9.]+)/([0-9.]+)"
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


def parse_log_metrics(path: Path) -> Optional[Dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    best_acc = None
    best_epoch = None
    total_epochs = None
    best_val_loss = None
    for match in EPOCH_PATTERN.finditer(text):
        epoch = int(match.group(1))
        total_epochs = int(match.group(2))
        val_loss = float(match.group(5))
        val_acc = float(match.group(6))
        if best_acc is None or val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_val_loss = val_loss
    if best_acc is None:
        return None
    return {
        "best_epoch": best_epoch,
        "total_epochs": total_epochs,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_acc,
    }


def collect_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for log_path in sorted(LOG_ROOT.rglob("*.log")):
        model, params = parse_params(log_path)
        metrics = parse_log_metrics(log_path)
        if metrics is None:
            print(f"Skipping {log_path}: no epoch summaries found.")
            continue
        relative_path = log_path.relative_to(PROJECT_ROOT)
        row = {
            "model": model,
            "log_path": str(relative_path).replace("\\", "/"),
            "params": params,
            **metrics,
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
