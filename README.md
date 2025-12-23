# Plant Seedlings Classification

## 问题描述

本项目为 Kaggle 竞赛 "Plant Seedlings Classification" 的解决方案，旨在根据植物幼苗的图像对其进行分类。数据集包含 12 类不同植物幼苗的高分辨率图像，挑战在于处理复杂背景、光照变化及类间相似性。具体题目可见 [Problem 页面](./problem.md)。

## 数据集

数据集可从 Kaggle 竞赛页面下载：[Plant Seedlings Classification](https://www.kaggle.com/competitions/plant-seedlings-classification/overview)。下载后请将数据解压至 `data/` 目录，确保包含以下结构：

```
data/
├── train/
│   ├── Black-grass/
│   ├── Charlock/
│   ├── Cleavers/
│   ├── Common Chickweed/
│   ├── Common wheat/
│   ├── Fat Hen/
│   ├── Loose Silky-bent/
│   ├── Maize/
│   ├── Scentless Mayweed/
│   ├── Shepherds Purse/
│   ├── Small-flowered Cranesbill/
│   └── Sugar beet/
├── test/
└── sample_submission.csv
```

## (一) 训练方法

### 1. 传统特征 + 经典模型
- **流程与原理**：将原始图像缩放到 256×256，提取 RGB 均值/方差、HSV 直方图、灰度直方图与 Laplacian/Sobel 纹理统计，使用 `StandardScaler` 归一化后喂入 SVM（RBF）或随机森林，实现经典机器学习基线。
- **运行方式**：
	```bash
	python src/baseline_features.py \
	    --data-root data \
	    --model svm \
	    --val-ratio 0.2 \
	    --seed 42 \
	    --output submissions/baseline.csv
	```

### 2. 自研轻量 CNN
- **流程与原理**：`SmallCNN` 采用轻量 Stem + 多个残差块，配合 Mixup、Label Smoothing、Cosine 学习率与可选强增强，实现从零训练的高效 CNN。
- **运行方式**：
	```bash
	python src/cnn_scratch.py \
	    --data-root data \
	    --epochs 40 \
	    --batch-size 64 \
	    --lr 3e-4 \
	    --mixup-alpha 0.2 \
	    --label-smoothing 0.1 \
	    --strong-aug \
	    --amp \
	    --predict-test \
	    --submission submissions/cnn.csv
	```

### 3. 预训练卷积模型
- **流程与原理**：基于 ImageNet 预训练的 ResNet / EfficientNet / ConvNeXt，先冻结骨干微调分类头，再解冻全网微调；可配合 AMP、强增强与 TTA 提升收敛速度与精度。
- **运行方式**：
	```bash
	python src/transfer_ensemble.py \
	    --mode train \
	    --data-root data \
	    --arch convnext_large \
	    --image-size 384 \
	    --epochs 12 \
	    --freeze-epochs 2 \
	    --batch-size 32 \
	    --lr 2e-4 \
	    --strong-aug \
	    --warmup-epochs 2 \
	    --grad-accum-steps 2 \
	    --tta 4 \
	    --amp \
	    --output experiments/convnext_large_best.pth \
	    --submission submissions/convnext_large.csv
	```

### 4. ViT 预训练 + 微调
- **流程与原理**：使用 `timm` 的 `vit_base_patch16_224.augreg_in21k` 或 `vit_large_patch16_224.augreg_in21k`，通过 Layer-wise LR Decay + Warmup + 强增强稳定微调；推理阶段结合 TTA。
- **运行方式**：
	```bash
	python src/transfer_ensemble.py \
	    --mode train \
	    --data-root data \
	    --arch vit_l_16 \
	    --epochs 8 \
	    --freeze-epochs 2 \
	    --batch-size 32 \
	    --lr 5e-5 \
	    --layer-decay 0.75 \
	    --strong-aug \
	    --warmup-epochs 1 \
	    --tta 4 \
	    --amp \
	    --output experiments/vit_l16_best.pth \
	    --submission submissions/vit_l16.csv
	```

### 5. CNN + ViT + 强增强 + 集成
- **流程与原理**：同时训练自研 CNN（提供纹理和局部敏感特征）与 ConvNeXt/Vision Transformer（捕获全局上下文），全部使用强增强和 TTA；最终以加权投票融合多个提交文件，缓解单模型偏差。
- **运行方式**：
	```bash
	# 训练自研 CNN 与 ViT，并生成提交
	python src/cnn_scratch.py \
	    --data-root data \
	    --epochs 40 \
	    --batch-size 64 \
	    --lr 3e-4 \
	    --mixup-alpha 0.2 \
	    --label-smoothing 0.1 \
	    --strong-aug \
	    --amp \
	    --predict-test \
	    --submission submissions/cnn.csv
	python src/transfer_ensemble.py \
	    --mode train \
	    --data-root data \
	    --arch vit_b_16 \
	    --epochs 40 \
	    --freeze-epochs 5 \
	    --batch-size 48 \
	    --lr 1e-4 \
	    --layer-decay 0.8 \
	    --strong-aug \
	    --tta 4 \
	    --amp \
	    --submission submissions/vit.csv

	# 加权集成提交结果
	python src/transfer_ensemble.py --mode ensemble \
	    --data-root data \
	    --ensemble-csvs submissions/cnn.csv submissions/vit.csv submissions/convnext_large.csv \
	    --ensemble-weights 1.0 1.2 1.0 \
	    --ensemble-output submissions/ensemble.csv
	```

## (二) 脚本信息

### `src/baseline_features.py`
| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data-root` | `data` | 数据根目录，需包含 `train/`、`test/` 与 `sample_submission.csv` |
| `--model` | `svm` | `svm` / `rf`，分别对应 RBF SVM 与随机森林 |
| `--val-ratio` | `0.2` | 划分验证集比例 |
| `--seed` | `42` | 随机种子，控制划分与模型初始化 |
| `--output` | `submissions/baseline.csv` | 预测输出 CSV 路径 |
| `--sample-csv` | `data/sample_submission.csv` | Kaggle 官方示例，用于保持文件顺序 |
| `--skip-train` | `False` | 仅生成特征矩阵，跳过训练与推理 |

### `src/cnn_scratch.py`
| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data-root` | `data` | 数据根目录 |
| `--epochs` | `40` | 训练轮数 |
| `--batch-size` | `32` | 批大小 |
| `--lr` | `3e-4` | 初始学习率（AdamW） |
| `--weight-decay` | `1e-4` | L2 正则 |
| `--val-ratio` | `0.2` | 训练/验证划分比例 |
| `--image-size` | `224` | 输入分辨率 |
| `--seed` | `42` | 随机种子 |
| `--num-workers` | `4` | DataLoader 线程数 |
| `--label-smoothing` | `0.1` | 交叉熵标签平滑 |
| `--mixup-alpha` | `0.2` | Mixup Beta 分布参数，0 表示关闭 |
| `--amp` | `False` | 启用自动混合精度 |
| `--output` | `experiments/cnn_scratch_best.pth` | 最优模型保存位置 |
| `--predict-test` | `False` | 训练后对测试集推理并生成 CSV |
| `--submission` | `submissions/cnn.csv` | 推理输出文件 |
| `--strong-aug` | `False` | 启用 Albumentations 强增强 |

### `src/transfer_ensemble.py`
| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--mode` | `train` | `train` 训练单模型，`ensemble` 融合多个提交 |
| `--arch` | `resnet50` | 预训练架构（ResNet/EfficientNet/ConvNeXt/ViT） |
| `--data-root` | `data` | 数据根目录 |
| `--image-size` | `224` | 训练与推理输入尺寸 |
| `--epochs` | `30` | 总训练轮数 |
| `--freeze-epochs` | `1` | 仅训练分类头的轮数 |
| `--batch-size` | `32` | 批大小 |
| `--lr` | `1e-4` | 学习率（可与 layer decay 联动） |
| `--weight-decay` | `1e-4` | L2 正则 |
| `--val-ratio` | `0.2` | 验证比例 |
| `--num-workers` | `4` | DataLoader 线程数 |
| `--label-smoothing` | `0.05` | 交叉熵标签平滑 |
| `--tta` | `4` | 推理 TTA 次数（1/2/4） |
| `--amp` | `False` | 自动混合精度 |
| `--output` | `experiments/transfer_best.pth` | 模型保存路径 |
| `--submission` | `submissions/transfer.csv` | 推理输出 CSV |
| `--ensemble-output` | `submissions/ensemble.csv` | 集成模式输出 |
| `--ensemble-csvs` | `[]` | 需要加权投票的提交文件列表 |
| `--ensemble-weights` | `None` | 对应文件的浮点权重，缺省则均为 1 |
| `--seed` | `42` | 随机种子 |
| `--strong-aug` | `False` | Albumentations 强增强 |
| `--layer-decay` | `1.0` | ViT 层级学习率衰减（<1 时生效） |
| `--warmup-epochs` | `0` | 学习率 warmup 轮数 |
| `--grad-accum-steps` | `1` | 梯度累积步数，用于大批次模拟 |
