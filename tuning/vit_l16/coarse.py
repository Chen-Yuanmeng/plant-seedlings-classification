from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
TUNE_MODULE_PATH = PROJECT_ROOT / "tuning" / "tune_models.py"
MODEL_NAME = "vit_l16"
MODEL_SCRIPT = PROJECT_ROOT / "src" / "transfer_ensemble.py"
MODEL_DIR = PROJECT_ROOT / "experiments" / "tuning" / "vit_l16"
SUBMISSION_DIR = PROJECT_ROOT / "submissions" / "tuning" / "vit_l16"
LOG_DIR = PROJECT_ROOT / "reports" / "tuning_logs" / "vit_l16" / "coarse"
COARSE_RESULT = SCRIPT_DIR / "coarse-result.txt"

BASE_ARGS: Dict[str, object] = {
    "mode": "train",
    "data-root": "data",
    "arch": "vit_l_16",
    "image-size": 224,
    "epochs": 8,
    "val-ratio": 0.2,
    "num-workers": 4,
    "strong-aug": True,
    "amp": True,
    "remove-bg": False,
}

BASELINE_OVERRIDES: Dict[str, object] = {
    "lr": 5.5e-5,
    "layer-decay": 0.7,
    "batch-size": 24,
    "warmup-epochs": 1,
    "label-smoothing": 0.05,
    "tta": 4,
}

VARIATIONS: Dict[str, Iterable[object]] = {
    "lr": [3.5e-5, 4.5e-5, 5.5e-5, 6.5e-5],
    "layer-decay": [0.6, 0.7, 0.8],
    "batch-size": [16, 24, 32],
    "warmup-epochs": [1, 2],
    "label-smoothing": [0.05, 0.1],
}

LOG_KEYWORDS = ("best", "val", "acc", "score", "loss")


def load_tune_module():
    spec = importlib.util.spec_from_file_location("tune_models", TUNE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load tune_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def summarize_log(log_path: Path, max_lines: int = 5) -> List[str]:
    if not log_path.exists():
        return ["log missing"]
    lines = [line.rstrip() for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    highlights = [line for line in lines if any(keyword in line.lower() for keyword in LOG_KEYWORDS)]
    if not highlights:
        highlights = lines
    return highlights[-max_lines:]


def build_run_plan() -> List[Tuple[str, object, Dict[str, object]]]:
    plan = [("baseline", None, {})]
    for name, values in VARIATIONS.items():
        baseline_value = BASELINE_OVERRIDES[name]
        for value in values:
            if value == baseline_value:
                continue
            plan.append((name, value, {name: value}))
    return plan


def run_ofat():
    tune = load_tune_module()
    python_bin = tune.resolve_python()
    summaries = []
    for path in (MODEL_DIR, SUBMISSION_DIR, LOG_DIR):
        tune.ensure_parent(path / "placeholder")

    for label, value, overrides in build_run_plan():
        run_args = dict(BASE_ARGS)
        run_args.update(BASELINE_OVERRIDES)
        run_args.update(overrides)
        tag = tune.format_tag(overrides)
        model_path = MODEL_DIR / f"{MODEL_NAME}{tag}.pth"
        submission_path = SUBMISSION_DIR / f"{MODEL_NAME}{tag}.csv"
        log_path = LOG_DIR / f"{MODEL_NAME}{tag}.log"
        run_args["output"] = str(model_path)
        run_args["submission"] = str(submission_path)
        command = tune.build_command(python_bin, MODEL_SCRIPT, run_args)
        print(f"Running: {' '.join(command)}")
        tune.run_command(command, log_path)
        snippet = summarize_log(log_path)
        summaries.append(
            {
                "param": label,
                "value": value,
                "log": log_path,
                "snippet": snippet,
            }
        )

    write_summary(summaries)


def write_summary(summaries: List[Dict[str, object]]):
    lines: List[str] = []
    for idx, item in enumerate(summaries, start=1):
        param = item["param"]
        value = item["value"]
        log_path: Path = item["log"]  # type: ignore[assignment]
        rel_log = log_path.relative_to(PROJECT_ROOT)
        lines.append(f"Run {idx}: {param if param != 'baseline' else 'baseline'} -> {value}")
        lines.append(f"    log: {rel_log}")
        for snippet_line in item["snippet"]:  # type: ignore[index]
            lines.append(f"    {snippet_line}")
        lines.append("")
    COARSE_RESULT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_ofat()
