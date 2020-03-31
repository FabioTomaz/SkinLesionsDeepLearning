#! /bin/bash
set -euxo pipefail

# UNITS 512, L2, PATIENCE 20

# BATCH SIZE
MODELS=("DenseNet201" "DenseNet201" "DenseNet201" "DenseNet201")
BATCH=(8 16 32 64)
FEEPOCHS=(0 0 0 0)
FELR=(1e-4 1e-4 1e-4 1e-4)
FTLR=(1e-4 1e-4 1e-4 1e-4)

for i in "${!MODELS[@]}"; do 
    echo "Model: ${MODELS[$i]}\n" >> output.txt
    echo "Batch size: ${BATCH[$i]}\n" >> output.txt
    echo "Weight initialization epochs: ${FEEPOCHS[$i]}\n" >> output.txt
    echo "Weight initialization learning rate: ${FELR[$i]}\n" >> output.txt
    echo "Fine tuning learning rate: ${FTLR[$i]}\n" >> output.txt

    python3 main.py /home/fmts/msc/experiments/data/isic2019/sampled_unbalanced_5000/ \
        --training --predval \
        --batchsize "${BATCH[i]}" \
        --maxqueuesize 10 \
        --model "${MODELS[i]}" \
        --feepochs "${FEEPOCHS[i]}" \
        --felr "${FELR[i]}" \
        --ftlr "${FTLR[i]}"
done


# WEIGHT INITIALIZATION EPOCHS
MODELS=("DenseNet201" "DenseNet201" "DenseNet201" "DenseNet201" "DenseNet201")
BATCH=(16 16 16 16 16)
FEEPOCHS=(0 1 2 3 4)
FELR=(1e-4 1e-4 1e-4 1e-4 1e-4)
FTLR=(1e-4 1e-4 1e-4 1e-4 1e-4)

for i in "${!MODELS[@]}"; do 
    echo "Model: ${MODELS[$i]}\n" >> output.txt
    echo "Batch size: ${BATCH[$i]}\n" >> output.txt
    echo "Weight initialization epochs: ${FEEPOCHS[$i]}\n" >> output.txt
    echo "Weight initialization learning rate: ${FELR[$i]}\n" >> output.txt
    echo "Fine tuning learning rate: ${FTLR[$i]}\n" >> output.txt

    python3 main.py /home/fmts/msc/experiments/data/isic2019/sampled_unbalanced_5000/ \
        --training --predval \
        --batchsize "${BATCH[i]}" \
        --maxqueuesize 10 \
        --model "${MODELS[i]}" \
        --feepochs "${FEEPOCHS[i]}" \
        --felr "${FELR[i]}" \
        --ftlr "${FTLR[i]}"
done