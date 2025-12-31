"""植物幼苗分类 WebUI 演示，用于展示五种模型的单张推理结果。"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Sequence, Tuple

import gradio as gr
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import predict_single
DEFAULT_TEST_ROOT = PROJECT_ROOT / "data" / "test"
VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    model_type: str
    image_size: int
    tta: int
    checkpoint: str | None = None
    remove_bg: bool = False
    arch: str | None = None
    ensemble_types: Tuple[str, ...] | None = None
    ensemble_checkpoints: Tuple[str, ...] | None = None
    ensemble_weights: Tuple[float, ...] | None = None


MODEL_SPECS: Sequence[ModelSpec] = (
    ModelSpec(
        key="baseline",
        display_name="Baseline (SVM)",
        model_type="baseline",
        image_size=224,
        tta=1,
        remove_bg=True,
    ),
    ModelSpec(
        key="cnn",
        display_name="Custom CNN",
        model_type="cnn",
        image_size=224,
        tta=4,
    ),
    ModelSpec(
        key="pretrained_cnn",
        display_name="ConvNeXt Large",
        model_type="pretrained_cnn",
        image_size=384,
        tta=4,
    ),
    ModelSpec(
        key="vit",
        display_name="ViT-L/16",
        model_type="vit",
        image_size=224,
        tta=4,
    ),
    ModelSpec(
        key="ensemble",
        display_name="Ensemble (CNN + ConvNeXt + ViT)",
        model_type="ensemble",
        image_size=224,
        tta=4,
        ensemble_types=("cnn", "pretrained_cnn", "vit"),
        ensemble_checkpoints=(
            "experiments/cnn_scratch_best.pth",
            "experiments/convnext_large_best.pth",
            "experiments/vit_l16_best.pth",
        ),
        ensemble_weights=(1.0, 1.2, 1.0),
    ),
)


class DemoEngine:
    def __init__(self, test_root: Path, device_arg: str | None, bg_min_ratio: float) -> None:
        self.test_root = test_root
        self.bg_min_ratio = bg_min_ratio
        self.device = predict_single.resolve_device(device_arg)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.image_paths = self._collect_images()
        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found under {self.test_root}. Ensure Kaggle test data is available."
            )

    def _collect_images(self) -> List[Path]:
        if not self.test_root.exists():
            return []
        paths: List[Path] = []
        for path in self.test_root.glob("**/*"):
            if path.is_file() and path.suffix.lower() in VALID_SUFFIXES:
                paths.append(path)
        return sorted(paths)

    def _sample_image(self) -> Path:
        return random.choice(self.image_paths)

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    def _predict_with_spec(self, spec: ModelSpec, image_path: Path) -> List[str]:
        ensemble_types = list(spec.ensemble_types) if spec.ensemble_types else []
        ensemble_checkpoints = list(spec.ensemble_checkpoints) if spec.ensemble_checkpoints else []
        ensemble_weights = list(spec.ensemble_weights) if spec.ensemble_weights else None
        args = argparse.Namespace(
            image=str(image_path),
            model_type=spec.model_type,
            checkpoint=spec.checkpoint,
            image_size=spec.image_size,
            device=None,
            tta=spec.tta,
            remove_bg=spec.remove_bg,
            bg_min_ratio=self.bg_min_ratio,
            topk=5,
            ensemble_types=ensemble_types,
            ensemble_checkpoints=ensemble_checkpoints,
            ensemble_weights=ensemble_weights,
            arch=spec.arch,
        )
        try:
            if spec.model_type == "ensemble":
                probs, classes = predict_single.predict_ensemble(args, self.device)
            else:
                probs, classes = predict_single.dispatch_single_prediction(args, self.device)
            top_idx = int(np.argmax(probs))
            top_label = classes[top_idx]
            top_conf = float(probs[top_idx])
            notes = summarize_topk(classes, probs, topk=3)
            return [spec.display_name, top_label, f"{top_conf * 100:.2f}%", notes]
        except Exception as exc:  # pragma: no cover - surface errors in UI
            self.logger.exception("Prediction failed for %s", spec.key)
            return [spec.display_name, "Error", "-", str(exc)]

    def predict_random_image(self) -> tuple[Image.Image, str, List[List[str]]]:
        image_path = self._sample_image()
        display_image = Image.open(image_path).convert("RGB")
        table_rows = [self._predict_with_spec(spec, image_path) for spec in MODEL_SPECS]
        return display_image, self._relative_path(image_path), table_rows


def summarize_topk(classes: Sequence[str], probs: Sequence[float], topk: int) -> str:
    order = np.argsort(probs)[::-1][:topk]
    return ", ".join(f"{classes[idx]} ({probs[idx] * 100:.1f}%)" for idx in order)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Plant Seedlings WebUI demo")
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT, help="Directory with Kaggle test images")
    parser.add_argument("--device", type=str, default=None, help="Torch device override, e.g. cpu or cuda:0")
    parser.add_argument("--bg-min-ratio", type=float, default=0.05, help="Background removal threshold (kept for parity)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web server host")
    parser.add_argument("--port", type=int, default=7860, help="Web server port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio public sharing")
    return parser.parse_args()


def launch(engine: DemoEngine, host: str, port: int, share: bool) -> None:
    with gr.Blocks(title="Plant Seedlings Web Demo") as demo:
        gr.Markdown(
            """# 植物幼苗分类在线演示
自动从 `data/test` 随机抽取一张测试图，对比 Baseline、轻量 CNN、ConvNeXt、ViT 与 Ensemble 的预测结果。
点击 **再抽一张图片** 可重新抽样。
"""
        )
        image_component = gr.Image(label="随机测试图像", type="pil")
        path_component = gr.Textbox(label="图像路径", interactive=False)
        table_component = gr.Dataframe(
            headers=["模型", "预测类别", "置信度", "Top-3 / 状态"],
            datatype=["str", "str", "str", "str"],
            row_count=(len(MODEL_SPECS), "fixed"),
            wrap=True,
        )
        predict_button = gr.Button("再抽一张图片", variant="primary")

        def _predict_wrapper():
            image, rel_path, rows = engine.predict_random_image()
            return image, rel_path, rows

        demo.load(_predict_wrapper, inputs=None, outputs=[image_component, path_component, table_component], queue=False)
        predict_button.click(
            _predict_wrapper,
            inputs=None,
            outputs=[image_component, path_component, table_component],
            queue=False,
        )

    demo.queue(max_size=8).launch(server_name=host, server_port=port, share=share)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args()
    engine = DemoEngine(args.test_root, args.device, args.bg_min_ratio)
    launch(engine, args.host, args.port, args.share)


if __name__ == "__main__":
    main()
