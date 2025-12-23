"""预训练模型微调与结果集成脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm
from torchvision import models
import timm

# 支持 python src/transfer_ensemble.py 运行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data import DataConfig, create_dataloaders, create_test_loader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预训练模型与集成工具")
    parser.add_argument("--mode", choices=["train", "ensemble"], default="train", help="运行模式")
    parser.add_argument(
        "--arch",
        choices=[
            "resnet50",
            "efficientnet_b0",
            "efficientnet_v2_m",
            "convnext_tiny",
            "convnext_large",
            "vit_b_16",
            "vit_l_16",
        ],
        default="resnet50",
        help="模型架构",
    )
    parser.add_argument("--data-root", type=str, default="data", help="数据根目录")
    parser.add_argument("--image-size", type=int, default=224, help="输入尺寸")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--freeze-epochs", type=int, default=1, help="仅训练分类头的轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证比例")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 线程数")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="标签平滑系数")
    parser.add_argument("--tta", type=int, default=4, help="推理时的 TTA 次数 (1/2/4)")
    parser.add_argument("--amp", action="store_true", help="开启 AMP")
    parser.add_argument("--output", type=str, default="experiments/transfer_best.pth", help="模型保存路径")
    parser.add_argument("--submission", type=str, default="submissions/transfer.csv", help="推理提交文件")
    parser.add_argument("--ensemble-output", type=str, default="submissions/ensemble.csv", help="集成输出文件")
    parser.add_argument("--ensemble-csvs", nargs="*", default=[], help="待集成的提交文件列表")
    parser.add_argument("--ensemble-weights", nargs="*", type=float, default=None, help="对应提交的权重")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--strong-aug", action="store_true", help="启用强数据增强")
    parser.add_argument("--layer-decay", type=float, default=1.0, help="ViT 层级学习率衰减系数 (<1 有效)")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="学习率 warmup 轮数")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="梯度累积步数")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if arch == "efficientnet_v2_m":
        weights = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
        model = models.efficientnet_v2_m(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if arch == "convnext_large":
        weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        model = models.convnext_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if arch == "vit_b_16":
        model = timm.create_model("vit_base_patch16_224.augreg_in21k", pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    if arch == "vit_l_16":
        model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    raise ValueError(f"未知架构: {arch}")


def toggle_backbone(model: nn.Module, arch: str, trainable: bool) -> None:
    if arch.startswith("vit"):
        for name, param in model.named_parameters():
            if name.startswith("head"):
                param.requires_grad = True
            else:
                param.requires_grad = trainable
    else:
        for name, param in model.named_parameters():
            if "classifier" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = trainable


def _get_vit_layer_id(name: str, num_layers: int) -> int:
    if name.startswith("cls_token") or name.startswith("pos_embed") or name.startswith("patch_embed"):
        return 0
    if name.startswith("blocks"):
        try:
            block_id = int(name.split(".")[1])
        except (IndexError, ValueError):
            block_id = 0
        return block_id + 1
    return num_layers - 1


def _build_vit_optimizer(model: nn.Module, base_lr: float, weight_decay: float, layer_decay: float) -> AdamW:
    num_layers = len(getattr(model, "blocks", [])) + 2
    layer_scales = [layer_decay ** (num_layers - 1 - i) for i in range(num_layers)]
    no_decay: Set[str] = set()
    if hasattr(model, "no_weight_decay"):
        nd = model.no_weight_decay()
        if isinstance(nd, (list, tuple, set)):
            no_decay = set(nd)
    param_groups: Dict[tuple[int, str], Dict[str, object]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = _get_vit_layer_id(name, num_layers)
        decay_tag = "no_decay" if (name in no_decay or name.endswith("bias") or "norm" in name.lower()) else "decay"
        key = (layer_id, decay_tag)
        if key not in param_groups:
            param_groups[key] = {
                "params": [],
                "lr": base_lr * layer_scales[layer_id],
                "weight_decay": 0.0 if decay_tag == "no_decay" else weight_decay,
            }
        param_groups[key]["params"].append(param)
    return AdamW(list(param_groups.values()))


def _build_optimizer(model: nn.Module, args: argparse.Namespace) -> AdamW:
    if args.layer_decay < 1.0 and args.arch.startswith("vit") and hasattr(model, "blocks"):
        return _build_vit_optimizer(model, args.lr, args.weight_decay, args.layer_decay)
    return AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)


def _build_scheduler(optimizer: AdamW, args: argparse.Namespace) -> CosineAnnealingLR | SequentialLR:
    if args.warmup_epochs > 0 and args.warmup_epochs < args.epochs:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
    return CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))


def _optimizer_step(optimizer: AdamW, scaler: torch.amp.GradScaler | None) -> None:
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def train_mode(args: argparse.Namespace) -> None:
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

    model = build_model(args.arch, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = _build_optimizer(model, args)
    scheduler = _build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    grad_accum = max(1, args.grad_accum_steps)

    best_acc = 0.0
    best_state = None
    non_blocking = config.pin_memory
    for epoch in range(1, args.epochs + 1):
        trainable = epoch > args.freeze_epochs
        toggle_backbone(model, args.arch, trainable)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)
        pending_update = False
        for step, batch in enumerate(tqdm(train_loader, desc=f"训练 Epoch {epoch}", leave=False), start=1):
            images = batch["image"].to(device, non_blocking=non_blocking)
            labels = batch["label"].to(device, non_blocking=non_blocking)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            scaled_loss = loss / grad_accum
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            pending_update = True
            if step % grad_accum == 0:
                _optimizer_step(optimizer, scaler)
                pending_update = False
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        if pending_update:
            _optimizer_step(optimizer, scaler)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证", leave=False):
                images = batch["image"].to(device, non_blocking=non_blocking)
                labels = batch["label"].to(device, non_blocking=non_blocking)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step()
        print(
            f"Epoch {epoch}/{args.epochs} | Train {train_loss:.4f}/{train_acc:.4f} | "
            f"Val {val_loss:.4f}/{val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_acc": best_acc,
                "class_to_idx": class_to_idx,
                "arch": args.arch,
            }
            ckpt_path = Path(args.output)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt_path)
            print(f"保存最优模型 {ckpt_path}，Val Acc={best_acc:.4f}")

    if best_state is None:
        print("训练失败，未获得有效模型。")
        return
    model.load_state_dict(best_state["model"])
    predict_test(
        args,
        model,
        device,
        class_to_idx,
        num_workers=args.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
    )


@torch.no_grad()
def _tta_inference(model: nn.Module, images: torch.Tensor, tta: int) -> torch.Tensor:
    logits = model(images)
    if tta <= 1:
        return logits
    aug_logits = [logits]
    if tta >= 2:
        aug_logits.append(model(torch.flip(images, dims=[3])))  # 水平翻转
    if tta >= 4:
        aug_logits.append(model(torch.flip(images, dims=[2])))  # 垂直翻转
        aug_logits.append(model(torch.flip(images, dims=[2, 3])))
    return torch.stack(aug_logits, dim=0).mean(dim=0)


@torch.no_grad()
def predict_test(
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    class_to_idx: Dict[str, int],
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> None:
    data_root = Path(args.data_root)
    test_loader = create_test_loader(
        data_root,
        class_to_idx,
        image_size=args.image_size,
        batch_size=64,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
    )
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    preds: List[str] = []
    paths: List[str] = []
    model.eval()
    for batch in tqdm(test_loader, desc="测试推理", leave=False):
        images = batch["image"].to(device, non_blocking=pin_memory)
        logits = _tta_inference(model, images, args.tta)
        labels = logits.softmax(dim=1).argmax(dim=1).cpu().numpy()
        preds.extend(idx_to_class[idx] for idx in labels)
        paths.extend(batch["path"])
    df = pd.read_csv(data_root / "sample_submission.csv")
    mapping = {Path(p).name: pred for p, pred in zip(paths, preds)}
    df["species"] = df["file"].map(mapping)
    out_path = Path(args.submission)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"测试集提交文件已保存：{out_path}")


def ensemble_mode(args: argparse.Namespace) -> None:
    if not args.ensemble_csvs:
        raise ValueError("集成模式需要至少一个 CSV 路径")
    weights = args.ensemble_weights
    if weights is None:
        weights = [1.0] * len(args.ensemble_csvs)
    if len(weights) != len(args.ensemble_csvs):
        raise ValueError("权重数量需与 CSV 数量一致")
    # 读取并构建 file -> species
    submissions = [pd.read_csv(path) for path in args.ensemble_csvs]
    sample_df = pd.read_csv(Path(args.data_root) / "sample_submission.csv")
    final_species: List[str] = []
    for file_id in sample_df["file"]:
        votes: Dict[str, float] = {}
        for df, w in zip(submissions, weights):
            row = df[df["file"] == file_id]
            if row.empty:
                continue
            label = row.iloc[0]["species"]
            votes[label] = votes.get(label, 0.0) + w
        if not votes:
            fallback = submissions[0][submissions[0]["file"] == file_id]
            final_species.append(fallback.iloc[0]["species"])  # 默认回退到首个 CSV
        else:
            final_species.append(max(votes.items(), key=lambda kv: kv[1])[0])
    sample_df["species"] = final_species
    out_path = Path(args.ensemble_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(out_path, index=False)
    print(f"集成结果已写入 {out_path}")


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train_mode(args)
    else:
        ensemble_mode(args)


if __name__ == "__main__":
    main()