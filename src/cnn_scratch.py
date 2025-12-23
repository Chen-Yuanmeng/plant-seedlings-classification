"""自研轻量 CNN 训练脚本（中文注释）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

# 便于直接 python src/cnn_scratch.py 运行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data import DataConfig, create_dataloaders, create_test_loader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自研 CNN 训练入口")
    parser.add_argument("--data-root", type=str, default="data", help="数据根目录")
    parser.add_argument("--epochs", type=int, default=40, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--image-size", type=int, default=224, help="输入分辨率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 线程数")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="标签平滑系数")
    parser.add_argument("--mixup-alpha", type=float, default=0.2, help="Mixup 参数，0 表示关闭")
    parser.add_argument("--amp", action="store_true", help="开启自动混合精度")
    parser.add_argument("--output", type=str, default="experiments/cnn_scratch_best.pth", help="模型保存路径")
    parser.add_argument("--predict-test", action="store_true", help="训练结束后对测试集推理并生成 CSV")
    parser.add_argument("--submission", type=str, default="submissions/cnn.csv", help="推理输出文件")
    parser.add_argument("--strong-aug", action="store_true", help="启用更强数据增强 (Albumentations)")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class ResidualBlock(nn.Module):
    """简化残差块。"""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        residual = x
        x = F.silu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.silu(x + residual)


class StemBlock(nn.Module):
    """初始卷积，包含深度可分离卷积以减小参数量。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SmallCNN(nn.Module):
    """自研轻量 CNN。"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        widths = [48, 96, 192, 320]
        self.stem = StemBlock(3, widths[0])
        stages = []
        in_ch = widths[0]
        for idx, width in enumerate(widths):
            if idx > 0:
                stages.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, width, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(width),
                        nn.SiLU(),
                    )
                )
                in_ch = width
            stages.append(ResidualBlock(in_ch))
            stages.append(ResidualBlock(in_ch))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(widths[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


def mixup_data(inputs: torch.Tensor, targets: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)
    mixed = lam * inputs + (1 - lam) * inputs[index]
    target_a = targets
    target_b = targets[index]
    return mixed, target_a, target_b, lam


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    mixup_alpha: float,
    non_blocking: bool,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader, desc="训练", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=non_blocking)
        labels = batch["label"].to(device, non_blocking=non_blocking)
        optimizer.zero_grad()
        mixed, target_a, target_b, lam = mixup_data(images, labels, mixup_alpha)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            outputs = model(mixed)
            if lam == 1.0:
                loss = criterion(outputs, labels)
            else:
                loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        progress.set_postfix(loss=running_loss / total, acc=correct / total)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    non_blocking: bool,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="验证", leave=False):
        images = batch["image"].to(device, non_blocking=non_blocking)
        labels = batch["label"].to(device, non_blocking=non_blocking)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return running_loss / total, correct / total


def save_checkpoint(state: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


@torch.no_grad()
def predict_test(
    model: nn.Module,
    device: torch.device,
    data_root: Path,
    class_to_idx: Dict[str, int],
    submission_path: Path,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> None:
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    test_loader = create_test_loader(
        data_root,
        class_to_idx,
        image_size=224,
        batch_size=64,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
    )
    model.eval()
    preds = []
    paths = []
    for batch in tqdm(test_loader, desc="推理", leave=False):
        images = batch["image"].to(device, non_blocking=pin_memory)
        logits = model(images)
        labels = logits.argmax(dim=1).cpu().numpy()
        preds.extend([idx_to_class[i] for i in labels])
        paths.extend(batch["path"])
    df = pd.read_csv(data_root / "sample_submission.csv")
    mapping = {Path(p).name: pred for p, pred in zip(paths, preds)}
    df["species"] = df["file"].map(mapping)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(submission_path, index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pin_memory = device.type == "cuda"
    config = DataConfig(
        data_root=Path(args.data_root),
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else 2,
        strong_augment=args.strong_aug,
    )
    train_loader, val_loader, class_to_idx = create_dataloaders(config)
    num_classes = len(class_to_idx)

    model = SmallCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    best_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.mixup_alpha,
            config.pin_memory,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, config.pin_memory)
        scheduler.step()
        print(
            f"Epoch {epoch}/{args.epochs} | Train {train_loss:.4f}/{train_acc:.4f} | "
            f"Val {val_loss:.4f}/{val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": best_acc,
                "class_to_idx": class_to_idx,
            }
            save_checkpoint(best_state, Path(args.output))
            print(f"保存新的最好模型，准确率 {best_acc:.4f}")

    if best_state is None:
        print("未能保存模型，请检查训练过程。")
        return

    model.load_state_dict(best_state["model"])
    if args.predict_test:
        predict_test(
            model,
            device,
            Path(args.data_root),
            class_to_idx,
            Path(args.submission),
            num_workers=args.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
        )
        print(f"测试集预测已保存到 {args.submission}")


if __name__ == "__main__":
    main()