# ConvNeXt-Large 粗调方案

## 基线设置
- 训练命令：`python src/transfer_ensemble.py --mode train --arch convnext_large`，基础参数见 [README.md](README.md)。
- 公共配置：384×384 输入、12 epoch（前 2 轮仅训练分类头）、`strong-aug`、AMP、`grad-accum-steps=2`、`warmup-epochs=2`、`tta=4`，均取自 [tuning/tune_models.py](tuning/tune_models.py) 的 `base_args`。
- 评估：固定 `val-ratio=0.2` 的验证集准确率 + Kaggle 私榜表现，日志集中在 `reports/tuning_logs/convnext_large/`。

## 粗调参数与理由
针对大模型，我们一次性探索与收敛、泛化和推理相关的关键超参：

| 参数 | 搜索空间 | 选择理由 |
| --- | --- | --- |
| `lr` | {1e-4, 1.5e-4, 2e-4} | README 推荐 2e-4，但 384px 训练常需更保守的 1e-4；设置三挡以观察收敛速度与震荡风险。|
| `weight-decay` | {3e-5, 7.5e-5, 1.2e-4} | 轻度到中度正则化跨度，帮助判断是否需要较大 L2 抑制高容量模型过拟合。|
| `batch-size` | {8, 12, 16} | 显存紧张时需退到 8，并依赖 grad accumulation；对比不同批量的稳定性和吞吐，决定细调时固定哪一档。|
| `freeze-epochs` | {1, 2, 3} | 预训练卷积在冻结时间过短/过长会影响低层特征；粗调对比可指出最佳解冻节奏。|
| `tta` | {2, 4} | TTA 次数影响测试阶段推理时长，但也与验证表现相关；先确认其对准确率的敏感程度，再决定细调是否单独展开。|

## 输出与记录
- Sweep 产物写入 `experiments/tuning/convnext_large/`、`submissions/tuning/convnext_large/`，命名遵守 `tune_models.py` Tag 规则，以便自动追踪。
- 运行完粗调脚本后，在 `tuning/convnext_large/coarse-result.txt` 汇整：各参数对性能的影响、推荐的细调方向（例如固定最佳批次后继续围绕 lr/weight decay 加密搜索，或在 tta=2 的代价收益下探索更长 epoch）。
