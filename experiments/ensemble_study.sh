#! /bin/bash
set -euxo pipefail

python3 main.py /home/fmts/msc/experiments/data/isic2019/sampled_balanced_82400_300/ \
    --training --predtest --predtestresultfolder test_predict_results \
    --batchsize 16 \
    --maxqueuesize 10 \
    --model DenseNet201 \
    --feepochs 2 \
    --ftepochs 100 \
    --felr 1e-3 \
    --ftlr 1e-4 \
    --online-data-augmentation 1

python3 main.py /home/fmts/msc/experiments/data/isic2019/sampled_balanced_82400_300/ \
    --training --predtest --predtestresultfolder test_predict_results \
    --batchsize 16 \
    --maxqueuesize 10 \
    --model ResNet152 \
    --feepochs 2 \
    --ftepochs 100 \
    --felr 1e-3 \
    --ftlr 1e-4 \
    --online-data-augmentation 1

python3 main.py /home/fmts/msc/experiments/data/isic2019/sampled_balanced_82400_300/ \
    --training --predtest --predtestresultfolder test_predict_results \
    --batchsize 16 \
    --maxqueuesize 10 \
    --model EfficientNetB2 \
    --feepochs 2 \
    --ftepochs 100 \
    --felr 1e-3 \
    --ftlr 1e-4 \
    --online-data-augmentation 1