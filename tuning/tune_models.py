from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


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


def resolve_python() -> str:
    candidates = [
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "Scripts" / "python",
        PROJECT_ROOT / ".venv" / "bin" / "python",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return sys.executable


def to_sequence(grid: Dict[str, Iterable[object]]) -> List[Dict[str, object]]:
    if not grid:
        return [{}]
    keys = list(grid)
    values = [grid[k] for k in keys]
    combos = []
    for choice in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, choice)})
    return combos


def sanitize_value(value: object) -> str:
    text = str(value)
    return (
        text.replace("/", "-")
        .replace(" ", "")
        .replace(".", "p")
        .replace("-", "m")
    )


def format_tag(params: Dict[str, object]) -> str:
    if not params:
        return "default"
    parts = []
    for key in sorted(params):
        parts.append(f"{key}_{sanitize_value(params[key])}")
    return "__" + "__".join(parts)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_command(python_bin: str, script: Path, args_map: Dict[str, object]) -> List[str]:
    command: List[str] = [python_bin, str(script)]
    for key in sorted(args_map):
        value = args_map[key]
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            command.append(flag)
            command.extend(str(v) for v in value)
            continue
        command.extend([flag, str(value)])
    return command


def run_command(command: List[str], log_path: Path) -> None:
    ensure_parent(log_path)
    print(f"Running: {' '.join(command)}")
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed, see log {log_path}")


MODEL_SWEEPS = [
    # {
    #     "name": "cnn_scratch",
    #     "script": PROJECT_ROOT / "src" / "cnn_scratch.py",
    #     "base_args": {
    #         "data-root": "data",
    #         "epochs": 40,
    #         "val-ratio": 0.2,
    #         "image-size": 224,
    #         "seed": 42,
    #         "num-workers": 4,
    #         "label-smoothing": 0.1,
    #         "strong-aug": True,
    #         "amp": True,
    #         "predict-test": True,
    #         "remove-bg": False,
    #     },
    #     "param_grid": {
    #         "lr": [2.5e-4, 3e-4, 4e-4],
    #         "batch-size": [40, 48],
    #         "mixup-alpha": [0.0, 0.2],
    #     },
    #     "model_dir": PROJECT_ROOT / "experiments" / "tuning" / "cnn",
    #     "submission_dir": PROJECT_ROOT / "submissions" / "tuning" / "cnn",
    #     "log_dir": PROJECT_ROOT / "reports" / "tuning_logs" / "cnn",
    # },
    # {
    #     "name": "convnext_large",
    #     "script": PROJECT_ROOT / "src" / "transfer_ensemble.py",
    #     "base_args": {
    #         "mode": "train",
    #         "data-root": "data",
    #         "arch": "convnext_large",
    #         "image-size": 384,
    #         "epochs": 12,
    #         "freeze-epochs": 2,
    #         "val-ratio": 0.2,
    #         "num-workers": 4,
    #         "strong-aug": True,
    #         "warmup-epochs": 2,
    #         "grad-accum-steps": 2,
    #         "tta": 4,
    #         "amp": True,
    #         "remove-bg": False,
    #     },
    #     "param_grid": {
    #         "lr": [1.5e-4, 2e-4],
    #         "batch-size": [12, 16],
    #         "weight-decay": [5e-5, 1e-4],
    #     },
    #     "model_dir": PROJECT_ROOT / "experiments" / "tuning" / "convnext_large",
    #     "submission_dir": PROJECT_ROOT / "submissions" / "tuning" / "convnext_large",
    #     "log_dir": PROJECT_ROOT / "reports" / "tuning_logs" / "convnext_large",
    # },
    {
        "name": "vit_l16",
        "script": PROJECT_ROOT / "src" / "transfer_ensemble.py",
        "base_args": {
            "mode": "train",
            "data-root": "data",
            "arch": "vit_l_16",
            "image-size": 224,
            "epochs": 8,
            "freeze-epochs": 2,
            "val-ratio": 0.2,
            "num-workers": 4,
            "strong-aug": True,
            "warmup-epochs": 1,
            "tta": 4,
            "amp": True,
            "remove-bg": False,
        },
        "param_grid": {
            "lr": [4e-5, 5e-5, 6e-5],
            "batch-size": [24, 32],
            "layer-decay": [0.65, 0.75],
        },
        "model_dir": PROJECT_ROOT / "experiments" / "tuning" / "vit_l16",
        "submission_dir": PROJECT_ROOT / "submissions" / "tuning" / "vit_l16",
        "log_dir": PROJECT_ROOT / "reports" / "tuning_logs" / "vit_l16",
    },
]


def run_sweep(python_bin: str, sweep_cfg: Dict[str, object]) -> None:
    script = sweep_cfg["script"]
    base_args = dict(sweep_cfg["base_args"])
    param_grid = sweep_cfg["param_grid"]
    model_dir: Path = sweep_cfg["model_dir"]
    submission_dir: Path = sweep_cfg["submission_dir"]
    log_dir: Path = sweep_cfg["log_dir"]
    ensure_parent(model_dir / "dummy")
    ensure_parent(submission_dir / "dummy")
    ensure_parent(log_dir / "dummy")
    combos = to_sequence(param_grid)
    for params in combos:
        tag = format_tag(params)
        model_path = model_dir / f"{sweep_cfg['name']}{tag}.pth"
        submission_path = submission_dir / f"{sweep_cfg['name']}{tag}.csv"
        log_path = log_dir / f"{sweep_cfg['name']}{tag}.log"
        run_args = dict(base_args)
        run_args.update(params)
        run_args["output"] = str(model_path)
        run_args["submission"] = str(submission_path)
        command = build_command(python_bin, script, run_args)
        run_command(command, log_path)


def main() -> None:
    python_bin = resolve_python()
    length = len(MODEL_SWEEPS)
    for i in range(length):
        sweep_cfg = MODEL_SWEEPS[i]
        print(f"Starting sweep {i + 1}/{length}: {sweep_cfg['name']}")
        run_sweep(python_bin, sweep_cfg)
        print(f"Completed sweep {i + 1}/{length}: {sweep_cfg['name']}")

if __name__ == "__main__":
    main()
