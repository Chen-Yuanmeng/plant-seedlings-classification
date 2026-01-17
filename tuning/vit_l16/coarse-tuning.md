# ViT-L/16 粗调方案

## 基线设置
- 训练命令：`python src/transfer_ensemble.py --mode train --arch vit_l_16`；默认 flag 参考 [README.md](README.md)。
- 公共配置：224×224 输入、8 epoch（前 2 轮冻结 backbone）、`val-ratio=0.2`、`strong-aug`、`warmup-epochs=1`、AMP、`tta=4`，与 [tuning/tune_models.py](tuning/tune_models.py) 的 `base_args` 保持一致。
- 指标：使用同一验证划分计算准确率，并同步提交至 Kaggle 获取外部对比；日志归档到 `reports/tuning_logs/vit_l16/`。

## 粗调参数与理由
Vit-L 的性能对学习率调度、层级衰减与正则项高度敏感，因此粗调阶段直接覆盖以下维度：

| 参数 | 搜索空间 | 选择理由 |
| --- | --- | --- |
| `lr` | {3.5e-5, 4.5e-5, 5.5e-5, 6.5e-5} | README 推荐 5e-5；向上、向下扩展 1e-5 以捕捉稳定/快速两端，必要时再在细调阶段集中于最优区间。|
| `layer-decay` | {0.6, 0.7, 0.8} | 低 decay (0.6) 更强调高层学习率，高 decay (0.8) 更接近均匀微调；帮助判断是否需要层级更陡或更平缓的学习率。|
| `batch-size` | {16, 24, 32} | 224px 输入在不同显存配置下可行的三档设置，较小批量对正则化友好，较大批量提升吞吐。|
| `warmup-epochs` | {1, 2} | Warmup 太短易导致梯度震荡，稍长可提升稳定性；粗调对比其收益。|
| `label-smoothing` | {0.05, 0.1} | ViT 默认 0.05，但 README 中 conv 模型常用 0.1；对比不同平滑强度对 12 类任务的影响。|

## 输出与记录
- Sweep 产生的模型/提交流向 `experiments/tuning/vit_l16/` 与 `submissions/tuning/vit_l16/`，保持 `tune_models.py` 的 tag 方案。
- 粗调完成后将准确率、Kaggle 分数、以及对上述参数的敏感度汇总至 `tuning/vit_l16/coarse-result.txt`，为细调阶段（例如锁定最佳 layer decay 后继续细化 lr 或引入背景去除）提供依据。
