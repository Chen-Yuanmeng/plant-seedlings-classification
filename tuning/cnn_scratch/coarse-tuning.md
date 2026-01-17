# CNN_Scratch 粗调方案

## 基线设置
- 训练入口：`python src/cnn_scratch.py`；完整默认值列于 [README.md](README.md) 的参数表。
- 公共设置：40 epoch、224×224 输入、`val-ratio=0.2`、`seed=42`、Albumentations 强增强与 AMP 开启，其他默认项沿用 [tuning/tune_models.py](tuning/tune_models.py) 的 `base_args`。
- 评估：统一统计验证集准确率与对应 submission 的 Kaggle 得分；标准日志落地 `reports/tuning_logs/cnn/`。

## 粗调参数与理由
为了在细调前全面摸底最敏感的训练因素，本阶段一次性覆盖学习率、批大小、正则策略与预处理。搜索空间如下：

| 参数 | 搜索空间 | 选择理由 |
| --- | --- | --- |
| `lr` | {2e-4, 3e-4, 4e-4, 5e-4} | AdamW 小模型对初始 lr 极为敏感；在 README 推荐 3e-4 周围再向两侧扩展，用以观察欠/过拟合临界点。|
| `batch-size` | {32, 40, 48} | 224 输入下 GPU 可承受的三个档位；较小批次带来更强正则化，较大批次配合 AMP 能提升吞吐。|
| `mixup-alpha` | {0.0, 0.2, 0.4} | Mixup 强度影响模型对背景噪声的鲁棒性；将默认 0.2 上下扩展，衡量其与学习率的交互。|
| `label-smoothing` | {0.05, 0.1} | README 默认 0.1，但在 12 类问题上较小的平滑可能保留更多判别力；粗调阶段验证差异。|
| `remove-bg` | {False, True} | 背景去除开销低但可能丢失上下文；通过显式比较决定是否纳入细调。|

## 输出与记录
- 模型、提交分别写入 `experiments/tuning/cnn/` 与 `submissions/tuning/cnn/`，命名沿用 `tune_models.py` 生成的 tag。
- 完成 sweep 后在 `tuning/cnn_scratch/coarse-result.txt` 记录：各组合性能、对上述参数的敏感度排序、需要在细调阶段进一步放大的维度（例如在最优 lr 周围加密间隔或引入 weight decay)。
