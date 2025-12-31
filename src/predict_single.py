"""Single-image inference helper covering all five training paradigms.

This script loads the trained checkpoints saved under ``experiments/`` and runs
prediction for a single image. It supports the five methods listed in the
README:

1. baseline          -> traditional features + classic ML
2. cnn               -> custom lightweight CNN trained from scratch
3. pretrained_cnn    -> ConvNeXt / EfficientNet style transfer learning
4. vit               -> ViT fine-tuning checkpoints
5. ensemble          -> weighted fusion of multiple checkpoints (above)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from joblib import load as joblib_load
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.background import remove_background  # noqa: E402
from src.baseline_features import extract_features  # noqa: E402
from src.cnn_scratch import SmallCNN  # noqa: E402
from src.transfer_ensemble import build_model as build_transfer_model  # noqa: E402

MODEL_CHOICES = ["baseline", "cnn", "pretrained_cnn", "vit", "ensemble"]
DEFAULT_CHECKPOINTS = {
    "baseline": "experiments/baseline_svm.joblib",
    "cnn": "experiments/cnn_scratch_best.pth",
    "pretrained_cnn": "experiments/convnext_large_best.pth",
    "vit": "experiments/vit_l16_best.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a single image with trained checkpoints")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument(
        "--model-type",
        choices=MODEL_CHOICES,
        required=True,
        help="Which training paradigm to use (see README)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for non-ensemble runs; defaults to experiments/* files",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Resize edge used for deep models")
    parser.add_argument("--device", type=str, default=None, help="Explicit torch device, e.g. cpu or cuda:0")
    parser.add_argument("--tta", type=int, default=1, help="Flip-based TTA for deep models (1/2/4)")
    parser.add_argument("--remove-bg", action="store_true", help="Apply color mask before inference")
    parser.add_argument(
        "--bg-min-ratio",
        type=float,
        default=0.05,
        help="Minimum plant ratio to keep the mask (same as training scripts)",
    )
    parser.add_argument("--topk", type=int, default=5, help="Number of predictions to display")
    parser.add_argument(
        "--ensemble-types",
        nargs="*",
        default=[],
        help="Model types participating in the ensemble (baseline/cnn/pretrained_cnn/vit)",
    )
    parser.add_argument(
        "--ensemble-checkpoints",
        nargs="*",
        default=[],
        help="Checkpoint list aligned with --ensemble-types",
    )
    parser.add_argument(
        "--ensemble-weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional weights aligned with ensemble members",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Override architecture when loading transfer checkpoints (rarely needed)",
    )
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint(path_str: str | None, model_type: str) -> Path:
    if model_type == "ensemble":
        raise ValueError("resolve_checkpoint should not be called for ensemble mode")
    default = DEFAULT_CHECKPOINTS.get(model_type)
    if path_str is None:
        if default is None:
            raise ValueError(f"No default checkpoint for model type {model_type}")
        path = Path(default)
    else:
        path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_pil_image(path: Path, *, remove_bg: bool, bg_min_ratio: float) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if not remove_bg:
        return image
    image_np = np.array(image)
    mask_result = remove_background(image_np, color_space="rgb", min_ratio=bg_min_ratio)
    rgb_image = cv2.cvtColor(mask_result.image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def tensorize_image(
    image_path: Path,
    image_size: int,
    remove_bg: bool,
    bg_min_ratio: float,
) -> torch.Tensor:
    image = load_pil_image(image_path, remove_bg=remove_bg, bg_min_ratio=bg_min_ratio)
    transform = build_eval_transform(image_size)
    tensor = transform(image)
    return tensor.unsqueeze(0)


def classes_from_mapping(class_to_idx: Dict[str, int]) -> List[str]:
    return [cls for cls, idx in sorted(class_to_idx.items(), key=lambda kv: kv[1])]


def tta_logits(model: torch.nn.Module, tensor: torch.Tensor, tta: int) -> torch.Tensor:
    logits = model(tensor)
    if tta <= 1:
        return logits
    augmented = [logits]
    if tta >= 2:
        augmented.append(model(torch.flip(tensor, dims=[3])))
    if tta >= 4:
        augmented.append(model(torch.flip(tensor, dims=[2])))
        augmented.append(model(torch.flip(tensor, dims=[2, 3])))
    return torch.stack(augmented, dim=0).mean(dim=0)


def predict_baseline(
    image_path: Path,
    checkpoint: Path,
    remove_bg: bool,
    bg_min_ratio: float,
) -> Tuple[np.ndarray, List[str]]:
    payload = joblib_load(checkpoint)
    model = payload["model"]
    scaler = payload["scaler"]
    features = extract_features(image_path, remove_bg=remove_bg, bg_min_ratio=bg_min_ratio)
    scaled = scaler.transform(features.reshape(1, -1))
    probs = model.predict_proba(scaled)[0]
    classes = list(model.classes_)
    return probs.astype(np.float64), classes


def predict_cnn(
    image_path: Path,
    checkpoint: Path,
    device: torch.device,
    image_size: int,
    remove_bg: bool,
    bg_min_ratio: float,
    tta: int,
) -> Tuple[np.ndarray, List[str]]:
    data = torch.load(checkpoint, map_location=device)
    class_to_idx = data["class_to_idx"]
    model = SmallCNN(len(class_to_idx)).to(device)
    model.load_state_dict(data["model"])
    model.eval()
    tensor = tensorize_image(image_path, image_size, remove_bg, bg_min_ratio).to(device)
    with torch.no_grad():
        logits = tta_logits(model, tensor, tta)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs, classes_from_mapping(class_to_idx)


def predict_transfer(
    image_path: Path,
    checkpoint: Path,
    device: torch.device,
    image_size: int,
    remove_bg: bool,
    bg_min_ratio: float,
    tta: int,
    arch_override: str | None = None,
) -> Tuple[np.ndarray, List[str]]:
    data = torch.load(checkpoint, map_location=device)
    class_to_idx = data["class_to_idx"]
    arch = data.get("arch") or arch_override
    if arch is None:
        raise ValueError("Checkpoint missing arch metadata; pass --arch explicitly")
    model = build_transfer_model(arch, len(class_to_idx)).to(device)
    model.load_state_dict(data["model"])
    model.eval()
    tensor = tensorize_image(image_path, image_size, remove_bg, bg_min_ratio).to(device)
    with torch.no_grad():
        logits = tta_logits(model, tensor, tta)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs, classes_from_mapping(class_to_idx)


def align_probabilities(
    probs: np.ndarray,
    source_classes: Sequence[str],
    target_order: Sequence[str],
) -> np.ndarray:
    lookup = {cls: idx for idx, cls in enumerate(source_classes)}
    return np.array([probs[lookup[cls]] for cls in target_order], dtype=np.float64)


def dispatch_single_prediction(args: argparse.Namespace, device: torch.device) -> Tuple[np.ndarray, List[str]]:
    image_path = Path(args.image)
    if args.model_type == "baseline":
        checkpoint = resolve_checkpoint(args.checkpoint, "baseline")
        return predict_baseline(image_path, checkpoint, args.remove_bg, args.bg_min_ratio)
    if args.model_type == "cnn":
        checkpoint = resolve_checkpoint(args.checkpoint, "cnn")
        return predict_cnn(
            image_path,
            checkpoint,
            device,
            args.image_size,
            args.remove_bg,
            args.bg_min_ratio,
            args.tta,
        )
    if args.model_type == "pretrained_cnn":
        checkpoint = resolve_checkpoint(args.checkpoint, "pretrained_cnn")
        return predict_transfer(
            image_path,
            checkpoint,
            device,
            args.image_size,
            args.remove_bg,
            args.bg_min_ratio,
            args.tta,
            arch_override=args.arch,
        )
    if args.model_type == "vit":
        checkpoint = resolve_checkpoint(args.checkpoint, "vit")
        return predict_transfer(
            image_path,
            checkpoint,
            device,
            args.image_size,
            args.remove_bg,
            args.bg_min_ratio,
            args.tta,
            arch_override=args.arch,
        )
    raise ValueError("Ensemble predictions use predict_ensemble() instead")


def predict_ensemble(args: argparse.Namespace, device: torch.device) -> Tuple[np.ndarray, List[str]]:
    types = args.ensemble_types or ["cnn", "pretrained_cnn", "vit"]
    checkpoints = args.ensemble_checkpoints or [DEFAULT_CHECKPOINTS[t] for t in types]
    if len(types) != len(checkpoints):
        raise ValueError("--ensemble-types and --ensemble-checkpoints must align")
    weights = args.ensemble_weights or [1.0] * len(types)
    if len(weights) != len(types):
        raise ValueError("Weights must align with ensemble members")

    aggregated: np.ndarray | None = None
    reference_classes: List[str] | None = None
    total_weight = 0.0
    for model_type, ckpt_str, weight in zip(types, checkpoints, weights):
        args_clone = argparse.Namespace(**vars(args))
        args_clone.model_type = model_type
        args_clone.checkpoint = ckpt_str
        probs, classes = dispatch_single_prediction(args_clone, device)
        if reference_classes is None:
            reference_classes = list(classes)
            aggregated = probs * weight
        else:
            aligned = align_probabilities(probs, classes, reference_classes)
            aggregated = aggregated + aligned * weight
        total_weight += weight
    assert aggregated is not None and reference_classes is not None
    if total_weight <= 0:
        raise ValueError("Sum of ensemble weights must be positive")
    aggregated /= total_weight
    return aggregated, reference_classes


def display_predictions(classes: Sequence[str], probs: Sequence[float], topk: int) -> None:
    order = np.argsort(probs)[::-1][:topk]
    print("Top predictions:")
    for rank, idx in enumerate(order, start=1):
        print(f"  {rank}. {classes[idx]:25s} {probs[idx]*100:6.2f}%")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.model_type == "ensemble":
        probs, classes = predict_ensemble(args, device)
    else:
        probs, classes = dispatch_single_prediction(args, device)
    display_predictions(classes, probs, args.topk)


if __name__ == "__main__":
    main()
