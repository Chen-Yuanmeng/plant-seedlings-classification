PYTHON=./.venv/Scripts/python

echo "模型 1: 传统特征 经典模型 (Baseline features)"

$PYTHON src/baseline_features.py \
    --data-root data \
    --model svm \
    --val-ratio 0.2 \
    --seed 42 \
    --remove-bg \
    --output submissions/baseline.csv

echo "模型 1 完成"
echo "==============================================="

echo "模型 2: 轻量 CNN"

$PYTHON src/cnn_scratch.py \
    --data-root data \
    --epochs 40 \
    --batch-size 40 \
    --lr 0.0003 \
    --mixup-alpha 0 \
    --label-smoothing 0.1 \
    --strong-aug \
    --amp \
    --predict-test \
    --submission submissions/cnn.csv \
    2>&1

echo "模型 2 完成"
echo "==============================================="

echo "模型 3: 预训练卷积模型"

$PYTHON src/transfer_ensemble.py \
    --mode train \
    --data-root data \
    --arch convnext_large \
    --image-size 384 \
    --epochs 12 \
    --freeze-epochs 2 \
    --batch-size 12 \
    --lr 0.00015 \
    --weight-decay 0.0001 \
    --strong-aug \
    --warmup-epochs 2 \
    --grad-accum-steps 2 \
    --tta 4 \
    --amp \
    --output experiments/convnext_large_best.pth \
    --submission submissions/convnext_large.csv \
    2>&1


echo "模型 3 完成"
echo "==============================================="

echo "模型 4: 预训练视觉 Transformer 模型 (ViT)"

$PYTHON src/transfer_ensemble.py \
    --mode train \
    --data-root data \
    --arch vit_l_16 \
    --epochs 8 \
    --freeze-epochs 2 \
    --batch-size 24 \
    --lr 5e-5 \
    --layer-decay 0.75 \
    --strong-aug \
    --warmup-epochs 1 \
    --tta 4 \
    --amp \
    --output experiments/vit_l16_best.pth \
    --submission submissions/vit_l16.csv \
    2>&1

echo "模型 4 完成"
echo "==============================================="

echo "模型 5: CNN + ViT + 强增强 + 集成"

$PYTHON src/transfer_ensemble.py --mode ensemble \
    --data-root data \
    --ensemble-csvs submissions/cnn.csv submissions/vit_l16.csv submissions/convnext_large.csv \
    --ensemble-weights 1.0 1.2 1.0 \
    --ensemble-output submissions/ensemble.csv \
    2>&1

echo "模型 5 完成"
echo "==============================================="
echo "全部实验完成"


