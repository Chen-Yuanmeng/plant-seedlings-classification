from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
TUNE_MODULE_PATH = PROJECT_ROOT / "tuning" / "tune_models.py"
MODEL_NAME = "vit_l16"
MODEL_SCRIPT = PROJECT_ROOT / "src" / "transfer_ensemble.py"
MODEL_DIR = PROJECT_ROOT / "experiments" / "tuning" / "vit_l16"
SUBMISSION_DIR = PROJECT_ROOT / "submissions" / "tuning" / "vit_l16"
LOG_DIR = PROJECT_ROOT / "reports" / "tuning_logs" / "vit_l16" / "fine"
FINE_RESULT = SCRIPT_DIR / "fine-result.txt"

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
    "tta": 4,
}

FIXED_ARGS: Dict[str, object] = {
    "batch-size": 16,
    "warmup-epochs": 1,
}

PARAM_GRID: Dict[str, Iterable[object]] = {
    "lr": [5e-5 + i * 5e-6 for i in range(0, 5)],
    "layer-decay": [0.65 + i * 0.05 for i in range(0, 4)],
    "label-smoothing": [0.025 + i * 0.025 for i in range(0, 3)],
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


def cleanup_files(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def artifacts_complete(paths: Iterable[Path]) -> bool:
    return all(path.exists() for path in paths)


def current_time_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_grid():
    tune = load_tune_module()
    python_bin = tune.resolve_python()
    for path in (MODEL_DIR, SUBMISSION_DIR, LOG_DIR):
        tune.ensure_parent(path / "placeholder")

    combos = tune.to_sequence(PARAM_GRID)
    total = len(combos)
    summaries: List[Dict[str, object]] = []

    for idx, params in enumerate(combos, start=1):
        run_args = dict(BASE_ARGS)
        run_args.update(FIXED_ARGS)
        run_args.update(params)
        tag = tune.format_tag(params)
        model_path = MODEL_DIR / f"{MODEL_NAME}{tag}.pth"
        submission_path = SUBMISSION_DIR / f"{MODEL_NAME}{tag}.csv"
        log_path = LOG_DIR / f"{MODEL_NAME}{tag}.log"
        run_args["output"] = str(model_path)
        run_args["submission"] = str(submission_path)
        command = tune.build_command(python_bin, MODEL_SCRIPT, run_args)

        artifacts = (model_path, submission_path, log_path)
        if artifacts_complete(artifacts):
            print(f"[{idx}/{total}] Skipping completed run: {tag}")
            snippet = ["skipped: artifacts already exist", *summarize_log(log_path)]
        else:
            if any(path.exists() for path in artifacts):
                print(f"[{idx}/{total}] Cleaning partial artifacts before rerun: {tag}")
                cleanup_files(artifacts)
            print(f"[{idx}/{total}] [{current_time_str()}] Running configuration {tag}")
            try:
                tune.run_command(command, log_path)
            except KeyboardInterrupt:
                cleanup_files((model_path, log_path))
                print(f"[{idx}/{total}] Interrupted. Removed partial model/log for: {tag}")
                raise
            snippet = summarize_log(log_path)
        summaries.append(
            {
                "params": params,
                "log": log_path,
                "snippet": snippet,
            }
        )

    write_summary(summaries)


def write_summary(summaries: List[Dict[str, object]]):
    lines: List[str] = []
    for idx, item in enumerate(summaries, start=1):
        params: Dict[str, object] = item["params"]  # type: ignore[assignment]
        parts = ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
        log_path: Path = item["log"]  # type: ignore[assignment]
        rel_log = log_path.relative_to(PROJECT_ROOT)
        lines.append(f"Run {idx}: {parts}")
        lines.append(f"    log: {rel_log}")
        for snippet_line in item["snippet"]:  # type: ignore[index]
            lines.append(f"    {snippet_line}")
        lines.append("")
    FINE_RESULT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_grid()
