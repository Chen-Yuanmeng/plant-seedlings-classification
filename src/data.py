"""数据加载与统计模块（中文注释要求）。"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class DataConfig:
    """数据与训练的基本配置。"""

    data_root: Path
    image_size: int = 224
    val_ratio: float = 0.2
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    use_stratified_split: bool = True
    pin_memory: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 2
    strong_augment: bool = False


class AlbumentationsWrapper:
    """包装 Albumentations 以兼容多进程 DataLoader。"""

    def __init__(self, augmentations: A.Compose) -> None:
        self.augmentations = augmentations
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image_np = np.array(image)
        transformed = self.augmentations(image=image_np)
        tensor = transformed["image"]
        if tensor.dtype != torch.float32:
            tensor = tensor.float() / 255.0
        return self.normalize(tensor)


class PlantSeedlingsDataset(Dataset):
    """通用数据集封装，兼容 train/val/test。"""

    def __init__(
        self,
        image_paths: Sequence[Path],
        class_to_idx: Dict[str, int],
        labels: Optional[Sequence[str]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.image_paths = list(image_paths)
        self.class_to_idx = class_to_idx
        self.labels = list(labels) if labels is not None else None
        self.transform = transform

    def __len__(self) -> int:  # noqa: D401
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        label = -1
        if self.labels is not None:
            label = self.class_to_idx[self.labels[idx]]
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long) if label != -1 else torch.tensor(-1),
            "path": str(path),
        }


def _standard_train_transform(image_size: int) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.2)),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _strong_train_transform(image_size: int) -> Callable:
    aug = A.Compose(
        [
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.5, 1.0), ratio=(0.75, 1.33)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.1, 0.1),
                rotate=(-25, 25),
                p=0.7,
            ),
            A.OneOf(
                [
                    A.ColorJitter(0.2, 0.2, 0.2, 0.1),
                    A.HueSaturationValue(15, 20, 15),
                    A.RandomBrightnessContrast(0.2, 0.2),
                ],
                p=0.9,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=5),
                ],
                p=0.1,
            ),
            A.CLAHE(p=0.1),
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(0.02, 0.12),
                hole_width_range=(0.02, 0.12),
                fill=0.0,
                p=0.5,
            ),
            ToTensorV2(),
        ]
    )
    return AlbumentationsWrapper(aug)


def _eval_transform(image_size: int) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _collect_images(train_dir: Path) -> Tuple[List[Path], List[str], Dict[str, int]]:
    """扫描训练集目录，返回路径和标签。"""

    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    image_paths: List[Path] = []
    labels: List[str] = []
    for cls in classes:
        for img_path in (train_dir / cls).glob("*.*"):
            image_paths.append(img_path)
            labels.append(cls)
    return image_paths, labels, class_to_idx


def create_dataloaders(config: DataConfig) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """构建 train/val dataloader，并返回类到索引的映射。"""

    train_dir = config.data_root / "train"
    image_paths, labels, class_to_idx = _collect_images(train_dir)
    if config.use_stratified_split:
        train_idx, val_idx = _stratified_split(labels, config.val_ratio, config.seed)
    else:
        train_idx, val_idx = _random_split(len(image_paths), config.val_ratio, config.seed)

    train_images = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_images = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_transform = _strong_train_transform(config.image_size) if config.strong_augment else _standard_train_transform(config.image_size)
    val_transform = _eval_transform(config.image_size)

    train_ds = PlantSeedlingsDataset(
        train_images,
        class_to_idx,
        train_labels,
        transform=train_transform,
    )
    val_ds = PlantSeedlingsDataset(
        val_images,
        class_to_idx,
        val_labels,
        transform=val_transform,
    )

    common_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }
    if config.num_workers > 0:
        common_kwargs["persistent_workers"] = config.persistent_workers
        common_kwargs["prefetch_factor"] = config.prefetch_factor

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **common_kwargs,
    )
    return train_loader, val_loader, class_to_idx


def create_test_loader(
    data_root: Path,
    class_to_idx: Dict[str, int],
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """构造测试集 dataloader。"""

    test_dir = data_root / "test"
    image_paths = sorted(test_dir.glob("*.*"))
    test_ds = PlantSeedlingsDataset(
        image_paths,
        class_to_idx,
        labels=None,
        transform=_eval_transform(image_size),
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(test_ds, **loader_kwargs)


def _stratified_split(labels: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """分层划分，保持类别比例。"""

    indices = np.arange(len(labels))
    stratify = np.array(labels)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify,
    )
    return train_idx.tolist(), val_idx.tolist()


def _random_split(num_samples: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """无分层随机划分。"""

    indices = list(range(num_samples))
    random.Random(seed).shuffle(indices)
    split = int(num_samples * (1 - val_ratio))
    return indices[:split], indices[split:]


def compute_class_weights(labels: Sequence[str]) -> torch.Tensor:
    """根据标签频率计算类别权重，便于处理类别不平衡。"""

    counter = Counter(labels)
    total = sum(counter.values())
    weights = [total / counter[cls] for cls in sorted(counter)]
    return torch.tensor(weights, dtype=torch.float32)


def summarize_dataset(data_root: Path) -> pd.DataFrame:
    """统计每个类别的样本量与示例分辨率，用于报告。"""

    train_dir = data_root / "train"
    rows: List[Dict[str, str]] = []
    for cls_dir in sorted(train_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        images = list(cls_dir.glob("*.*"))
        if not images:
            continue
        sample_image = Image.open(images[0]).size
        rows.append(
            {
                "class": cls_dir.name,
                "count": len(images),
                "width": sample_image[0],
                "height": sample_image[1],
            }
        )
    return pd.DataFrame(rows)


def create_folds(
    data_root: Path,
    n_splits: int = 5,
    seed: int = 42,
) -> List[pd.DataFrame]:
    """生成 K 折划分结果，便于提升最终精度。"""

    train_dir = data_root / "train"
    image_paths, labels, _ = _collect_images(train_dir)
    df = pd.DataFrame({"path": [str(p) for p in image_paths], "label": labels})
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: List[pd.DataFrame] = []
    for fold_id, (_, val_idx) in enumerate(skf.split(df["path"], df["label"])):
        fold_df = df.iloc[val_idx].copy()
        fold_df["fold"] = fold_id
        folds.append(fold_df)
    return folds
