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
        --remove-bg \
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
| `--remove-bg` | `False` | 是否移除背景 |
| `--bg-min-ratio` | `0.05` | 植物像素占比阈值，低于该值不使用掩码 |

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
| `--remove-bg` | `False` | 训练/推理阶段移除背景 |
| `--bg-min-ratio` | `0.05` | 背景去除的植物像素阈值 |

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
| `--remove-bg` | `False` | 训练/TTA 前对输入应用背景去除 |
| `--bg-min-ratio` | `0.05` | 背景掩码阈值，越大越保守 |

### `src/predict_single.py`

使用已经训练好的模型对单张图像进行预测，支持传统特征模型与深度学习模型。

```bash
# Baseline SVM features (defaults to experiments/baseline_svm.joblib)
python src/predict_single.py --image data/test/xxx.png --model-type baseline

# Small CNN with AMP-style TTA
python src/predict_single.py --image data/test/xxx.png --model-type cnn \
    --checkpoint experiments/cnn_scratch_best.pth --tta 4 --image-size 224

# ConvNeXt-L checkpoint at 384px
python src/predict_single.py --image data/test/xxx.png --model-type pretrained_cnn \
    --checkpoint experiments/convnext_large_best.pth --image-size 384 --tta 4

# Weighted ensemble of CNN + ConvNeXt + ViT
python src/predict_single.py --image data/test/xxx.png --model-type ensemble \
    --ensemble-types cnn pretrained_cnn vit \
    --ensemble-checkpoints experiments/cnn_scratch_best.pth \
                           experiments/convnext_large_best.pth \
                           experiments/vit_l16_best.pth \
    --ensemble-weights 1.0 1.2 1.0 --tta 4 --image-size 384
```

## (三) Web 演示

为了方便产品/业务侧快速体验单张样本推理，可通过 Gradio WebUI 随机抽取 `data/test` 中的图像，并同时查看 Baseline、自研轻量 CNN、预训练卷积模型、ViT 以及加权 Ensemble 的预测结果。所有 demo 代码位于 `demo/` 目录，旧版 `run_demo.py`、`run_demo.sh`、`runall.sh` 均已弃用。

```bash
python demo/webui.py \
    --device cuda:0 \
    --port 7860 \
    --share           # 仅在需要公网访问时添加
```

- 默认会自动检测 `data/test`，若需测试其他目录，可通过 `--test-root path/to/images` 指定。
- `--device` 可强制使用 `cpu` 或 `cuda:N`，否则脚本会自动检测可用 GPU。
- 点击界面中的 **再抽一张图片** 按钮可重新抽样测试图并刷新五个模型的预测与置信度。

## (四) 调参与自动化脚本

`tuning/` 目录提供了“粗调→细调”的两阶段流程，并且覆盖 CNN、ConvNeXt-L 微调方案以及 ViT-L/16：

- 粗调（`tuning/<model>/coarse.py`）：基于 One-Factor-at-a-Time (OFAT) 策略，每次只变动一个超参，方便判断哪些维度最敏感。运行脚本会调用 `tuning/tune_models.py` 的统一封装，在 `reports/tuning_logs/<model>/coarse/` 下写入日志，并把每次执行的关键信息总结到 `tuning/<model>/coarse-result.txt`。
- 细调（`tuning/<model>/fine.py`）：锁定粗调中表现稳定的超参后，对剩余敏感组合执行完整网格搜索，日志写到 `reports/tuning_logs/<model>/fine/`，执行摘要保存在 `tuning/<model>/fine-result.txt`。每个脚本会自动复用 `tune_models.py` 的标签生成/输出路径，确保实验、提交与日志按 `experiments/tuning/**`、`submissions/tuning/**` 的约定归档。
- 汇总：运行 `python tuning/summarize_tuning_logs.py` 会遍历 `reports/tuning_logs/**/fine/` 中的日志，解析最佳验证分数，输出 CSV 到 `reports/tuning_results/` 以及逐日志的 `reports/tuning_results/per_log/**`，便于在表格工具里进行多模型对比。

根据调参结果，选取最佳参数作为最终提交的配置。
