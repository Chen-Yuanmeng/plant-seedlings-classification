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

try:
    from .background import MaskResult, remove_background
except ImportError:  # pragma: no cover - fallback for script entry
    from background import MaskResult, remove_background


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="植物幼苗传统特征分类基线")
    parser.add_argument("--data-root", type=str, default="data", help="数据根目录")
    parser.add_argument("--model", type=str, choices=["svm", "rf"], default="svm", help="选择分类器类型")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default="submissions/baseline.csv", help="Kaggle 提交文件路径")
    parser.add_argument("--sample-csv", type=str, default="data/sample_submission.csv", help="示例提交文件用于确定顺序")
    parser.add_argument("--skip-train", action="store_true", help="仅生成特征不训练（调试用）")
    parser.add_argument("--remove-bg", action="store_true", help="是否去除背景")
    parser.add_argument(
        "--bg-min-ratio",
        type=float,
        default=0.05,
        help="植物像素占比阈值，小于该值时跳过掩码",
    )
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


def extract_features_with_bg_removal(path: Path, min_ratio: float = 0.05, verbose: bool = False) -> np.ndarray:
    """改进版：在植物区域上提取特征"""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    
    # 调整大小
    image = cv2.resize(image, (256, 256))

    mask_result: MaskResult = remove_background(image, color_space="bgr", min_ratio=min_ratio)
    plant_ratio = mask_result.ratio
    plant_image = mask_result.image
    mask = mask_result.mask
    use_mask = mask_result.used_mask
    if verbose and not use_mask:
        print(f"警告: {path.name} 植物区域过小 ({plant_ratio*100:.1f}%)，使用完整图像")
    
    # 在植物区域上计算特征
    rgb = cv2.cvtColor(plant_image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(plant_image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(plant_image, cv2.COLOR_BGR2GRAY)
    
    # 计算颜色特征
    if use_mask:
        # 只计算植物区域的统计量
        rgb_pixels = rgb[mask > 0]
        if len(rgb_pixels) > 0:
            rgb_mean = rgb_pixels.mean(axis=0)
            rgb_std = rgb_pixels.std(axis=0)
        else:
            rgb_mean = np.zeros(3)
            rgb_std = np.zeros(3)
    else:
        # 使用完整图像
        rgb_mean = rgb.mean(axis=(0, 1))
        rgb_std = rgb.std(axis=(0, 1))
    
    # 计算HSV直方图
    hsv_hist = []
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], mask if use_mask else None, [16], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hsv_hist.append(hist)
    
    # 计算灰度直方图
    gray_hist = cv2.calcHist([gray], [0], mask if use_mask else None, [32], [0, 256])
    gray_hist = cv2.normalize(gray_hist, gray_hist).flatten()
    
    # 计算纹理特征
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = laplacian.var()
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_stat = np.array([sobelx.mean(), sobelx.std(), sobely.mean(), sobely.std()])
    
    # 添加植物区域占比作为特征
    area_features = np.array([plant_ratio])
    
    features = np.concatenate(
        [
            rgb_mean,
            rgb_std,
            *hsv_hist,
            gray_hist,
            np.array([lap_var]),
            edge_stat,
            area_features,
        ]
    )
    return features.astype(np.float32)


def extract_features_original(path: Path) -> np.ndarray:
    """原始版本：在整个图像上提取特征"""
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


def extract_features(path: Path, remove_bg: bool = True, bg_min_ratio: float = 0.05) -> np.ndarray:
    """特征提取函数，根据参数选择是否去除背景"""
    if remove_bg:
        return extract_features_with_bg_removal(path, min_ratio=bg_min_ratio)
    else:
        return extract_features_original(path)


def build_feature_matrix(paths: Sequence[Path], remove_bg: bool = True, bg_min_ratio: float = 0.05) -> np.ndarray:
    feats = [
        extract_features(p, remove_bg, bg_min_ratio)
        for p in tqdm(paths, desc="提取特征", unit="img")
    ]
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


def visualize_mask_comparison(image_path: Path, save_path: Path = None):
    """可视化掩码效果，用于调试"""
    import matplotlib.pyplot as plt
    
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, (256, 256))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建掩码
    mask_result = remove_background(image, color_space="bgr", min_ratio=0.05)
    mask = mask_result.mask
    masked_image = cv2.cvtColor(mask_result.image, cv2.COLOR_BGR2RGB)
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(image_rgb)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 掩码
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('植物掩码')
    axes[1].axis('off')
    
    # 掩码后的图像
    axes[2].imshow(masked_image)
    axes[2].set_title('植物区域')
    axes[2].axis('off')
    
    plant_ratio = mask_result.ratio
    plt.suptitle(f'植物区域占比: {plant_ratio*100:.1f}%', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    
    return mask, plant_ratio


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    
    print(f"使用背景去除: {args.remove_bg}")
    
    # 调试：可视化一些样本的掩码效果
    if not args.skip_train:
        debug_samples = 0
        train_samples = list_train_images(train_dir)
        if len(train_samples) > 0:
            print(f"\n调试: 可视化前{debug_samples}个样本的掩码效果")
            for i, (img_path, label) in enumerate(train_samples[:debug_samples]):
                print(f"\n样本 {i+1}: {img_path.name} ({label})")
                mask, ratio = visualize_mask_comparison(img_path)
                if ratio < 0.05:
                    print(f"警告: 植物区域占比过低 ({ratio*100:.1f}%)")
    
    train_samples = list_train_images(train_dir)
    train_paths, train_labels = zip(*train_samples)
    
    # 构建特征矩阵
    X = build_feature_matrix(train_paths, remove_bg=args.remove_bg, bg_min_ratio=args.bg_min_ratio)
    y = np.array(train_labels)
     
    # 打印特征维度
    print(f"\n特征维度: {X.shape}")
    print(f"样本数量: {X.shape[0]}")
    print(f"特征数量: {X.shape[1]}")
    
    if args.skip_train:
        print("跳过训练，仅生成特征矩阵。")
        
        # 分析特征的统计信息
        print("\n特征统计信息:")
        print(f"特征均值范围: [{X.mean(axis=0).min():.4f}, {X.mean(axis=0).max():.4f}]")
        print(f"特征标准差范围: [{X.std(axis=0).min():.4f}, {X.std(axis=0).max():.4f}]")
        
        # 检查植物区域占比特征
        if args.remove_bg:
            plant_ratios = X[:, -1]  # 最后一个特征是植物区域占比
            print(f"\n植物区域占比统计:")
            print(f"最小值: {plant_ratios.min()*100:.2f}%")
            print(f"最大值: {plant_ratios.max()*100:.2f}%")
            print(f"平均值: {plant_ratios.mean()*100:.2f}%")
            print(f"小于5%的样本数: {np.sum(plant_ratios < 0.05)}")
            print(f"小于10%的样本数: {np.sum(plant_ratios < 0.1)}")
        
        return

    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=y,
    )

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 训练模型
    model = build_model(args.model, args.seed)
    model.fit(X_train_scaled, y_train)
    
    # 验证集评估
    val_pred = model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"\n验证集准确率: {val_acc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_val, val_pred))

    # 在整个训练集上重新训练
    full_scaler = StandardScaler().fit(X)
    X_full = full_scaler.transform(X)
    final_model = build_model(args.model, args.seed)
    final_model.fit(X_full, y)
    print(f"\n在整个训练集上的准确率: {final_model.score(X_full, y):.4f}")

    # 处理测试集
    test_paths = list_test_images(test_dir)
    if not test_paths:
        print("未找到测试集图像，跳过提交文件生成。")
        return

    # 提取测试集特征
    X_test = build_feature_matrix(test_paths, remove_bg=args.remove_bg, bg_min_ratio=args.bg_min_ratio)
    X_test_scaled = full_scaler.transform(X_test)
    
    # 预测
    test_pred = final_model.predict(X_test_scaled)
    
    # 生成提交文件
    sample_csv = pd.read_csv(args.sample_csv)
    id_to_pred = {p.name: cls for p, cls in zip(test_paths, test_pred)}
    sample_csv["species"] = sample_csv["file"].map(id_to_pred)
    
    sample_csv.to_csv(args.output, index=False)
    print(f"\n提交文件已保存: {args.output}")
    
    # 输出特征重要性（仅对随机森林）
    if args.model == "rf":
        importances = final_model.feature_importances_
        print(f"\nTop 10 重要特征:")
        indices = np.argsort(importances)[::-1][:10]
        for i, idx in enumerate(indices):
            feature_names = [
                "R_mean", "G_mean", "B_mean",
                "R_std", "G_std", "B_std",
                *[f"HSV_{i}_bin{j}" for i in range(3) for j in range(16)],
                *[f"Gray_bin{i}" for i in range(32)],
                "Laplacian_var",
                "SobelX_mean", "SobelX_std", "SobelY_mean", "SobelY_std",
                "Plant_ratio"
            ]
            print(f"  {i+1:2d}. {feature_names[idx]:20s} : {importances[idx]:.4f}")


if __name__ == "__main__":
    main()
