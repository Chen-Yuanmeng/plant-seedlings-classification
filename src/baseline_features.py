"""传统特征 + 经典机器学习基线。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="植物幼苗传统特征分类基线")
    parser.add_argument("--data-root", type=str, default="data", help="数据根目录")
    parser.add_argument("--model", type=str, choices=["svm", "rf"], default="svm", help="选择分类器类型")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default="submissions/baseline.csv", help="Kaggle 提交文件路径")
    parser.add_argument("--sample-csv", type=str, default="data/sample_submission.csv", help="示例提交文件用于确定顺序")
    parser.add_argument("--skip-train", action="store_true", help="仅生成特征不训练（调试用）")
    return parser.parse_args()


def list_train_images(train_dir: Path) -> List[Tuple[Path, str]]:
    samples: List[Tuple[Path, str]] = []
    for cls_dir in sorted(train_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img_path in cls_dir.glob("*.*"):
            samples.append((img_path, cls_dir.name))
    return samples


def list_test_images(test_dir: Path) -> List[Path]:
    return sorted(test_dir.glob("*.*"))


def extract_features(path: Path) -> np.ndarray:
    """为单张图片计算颜色/纹理特征。"""

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    image = cv2.resize(image, (256, 256))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rgb_mean = rgb.mean(axis=(0, 1))
    rgb_std = rgb.std(axis=(0, 1))

    hsv_hist = []
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [16], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hsv_hist.append(hist)

    gray_hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    gray_hist = cv2.normalize(gray_hist, gray_hist).flatten()

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = laplacian.var()

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_stat = np.array([sobelx.mean(), sobelx.std(), sobely.mean(), sobely.std()])

    features = np.concatenate(
        [
            rgb_mean,
            rgb_std,
            *hsv_hist,
            gray_hist,
            np.array([lap_var]),
            edge_stat,
        ]
    )
    return features.astype(np.float32)


def build_feature_matrix(paths: Sequence[Path]) -> np.ndarray:
    feats = [extract_features(p) for p in tqdm(paths, desc="提取特征", unit="img")]
    return np.stack(feats, axis=0)


def build_model(model_name: str, seed: int):
    if model_name == "svm":
        return SVC(C=10.0, kernel="rbf", gamma="scale", probability=True, random_state=seed)
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=4,
        n_jobs=-1,
        random_state=seed,
    )


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    train_samples = list_train_images(train_dir)
    train_paths, train_labels = zip(*train_samples)
    X = build_feature_matrix(train_paths)
    y = np.array(train_labels)

    if args.skip_train:
        print("跳过训练，仅生成特征矩阵。")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = build_model(args.model, args.seed)
    model.fit(X_train_scaled, y_train)
    val_pred = model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"验证集准确率: {val_acc:.4f}")
    print(classification_report(y_val, val_pred))

    full_scaler = StandardScaler().fit(X)
    X_full = full_scaler.transform(X)
    final_model = build_model(args.model, args.seed)
    final_model.fit(X_full, y)

    test_paths = list_test_images(test_dir)
    if not test_paths:
        print("未找到测试集图像，跳过提交文件生成。")
        return

    X_test = build_feature_matrix(test_paths)
    X_test_scaled = full_scaler.transform(X_test)
    test_pred = final_model.predict(X_test_scaled)

    sample_csv = pd.read_csv(args.sample_csv)
    id_to_pred = {p.name: cls for p, cls in zip(test_paths, test_pred)}
    sample_csv["species"] = sample_csv["file"].map(id_to_pred)
    sample_csv.to_csv(args.output, index=False)
    print(f"提交文件已保存: {args.output}")


if __name__ == "__main__":
    main()