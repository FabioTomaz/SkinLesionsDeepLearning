#! /bin/bash
set -euxo pipefail


# #BATCH SIZE
MODELS=("DenseNet201" "DenseNet201" "DenseNet201")
BATCH=(4 8 16 32)
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
        --training \
        --batchsize "${BATCH[i]}" \
        --maxqueuesize 10 \
        --model "${MODELS[i]}" \
        --feepochs "${FEEPOCHS[i]}" \
        --felr "${FELR[i]}" \
        --ftlr "${FTLR[i]}"
done


# # WEIGHT INITIALIZATION EPOCHS
MODELS=("DenseNet201" "DenseNet201" "DenseNet201" "DenseNet201" "DenseNet201")
BATCH=(16 16 16 16 16)
FEEPOCHS=(0 1 2 4 6)
FELR=(1e-4 1e-4 1e-4 1e-4 1e-4)
FTLR=(1e-4 1e-4 1e-4 1e-4 1e-4)

for i in "${!MODELS[@]}"; do 
    echo "Model: ${MODELS[$i]}\n" >> output.txt
    echo "Batch size: ${BATCH[$i]}\n" >> output.txt
    echo "Weight initialization epochs: ${FEEPOCHS[$i]}\n" >> output.txt
    echo "Weight initialization learning rate: ${FELR[$i]}\n" >> output.txt
    echo "Fine tuning learning rate: ${FTLR[$i]}\n" >> output.txt

    python3 main.py /home/fmts/msc/experiments/data/isic2019/sampled_unbalanced_5000/ \
        --training --predtest \
        --batchsize "${BATCH[i]}" \
        --maxqueuesize 10 \
        --model "${MODELS[i]}" \
        --feepochs "${FEEPOCHS[i]}" \
        --felr "${FELR[i]}" \
        --ftlr "${FTLR[i]}"
done

# # WEIGHT INITIALIZATION LR
MODELS=("DenseNet201" "DenseNet201" "DenseNet201")
BATCH=(16 16 16)
FEEPOCHS=(2 2 2)
FELR=(1e-3 1e-4 1e-5)
FTLR=(1e-4 1e-4 1e-4)

for i in "${!MODELS[@]}"; do 
    echo "Model: ${MODELS[$i]}\n" >> output.txt
    echo "Batch size: ${BATCH[$i]}\n" >> output.txt
    echo "Weight initialization epochs: ${FEEPOCHS[$i]}\n" >> output.txt
    echo "Weight initialization learning rate: ${FELR[$i]}\n" >> output.txt
    echo "Fine tuning learning rate: ${FTLR[$i]}\n" >> output.txt

    python3 main.py /home/fmts/msc/experiments/data/isic2019/sampled_unbalanced_5000/ \
        --training --predtest \
        --batchsize "${BATCH[i]}" \
        --maxqueuesize 10 \
        --model "${MODELS[i]}" \
        --feepochs "${FEEPOCHS[i]}" \
        --felr "${FELR[i]}" \
        --ftlr "${FTLR[i]}"
done

# # FINE TUNING LR
MODELS=("DenseNet201")
BATCH=(16 16 16 16)
FEEPOCHS=(2 2 2 2)
FELR=(1e-3 1e-3 1e-3 1e-3)
FTLR=(1e-4 1e-5 1e-4 1e-3)

for i in "${!MODELS[@]}"; do 
   echo "Model: ${MODELS[$i]}\n" >> output.txt
   echo "Batch size: ${BATCH[$i]}\n" >> output.txt
   echo "Weight initialization epochs: ${FEEPOCHS[$i]}\n" >> output.txt
   echo "Weight initialization learning rate: ${FELR[$i]}\n" >> output.txt
   echo "Fine tuning learning rate: ${FTLR[$i]}\n" >> output.txt

   python3 main.py /home/fmts/msc/experiments/data/isic2019/sampled_unbalanced_5000/ \
       --training --predtest \
       --batchsize "${BATCH[i]}" \
       --maxqueuesize 10 \
       --model "${MODELS[i]}" \
       --feepochs "${FEEPOCHS[i]}" \
       --felr "${FELR[i]}" \
       --ftlr "${FTLR[i]}"
done

# # DROPOUT RATE
MODELS=("DenseNet201" "DenseNet201" "DenseNet201" "DenseNet201" "DenseNet201")
BATCH=(16 16 16 16 16)
FEEPOCHS=(2 2 2 2 2)
FELR=(1e-3 1e-3 1e-3 1e-3 1e-3)
FTLR=(1e-4 1e-4 1e-4 1e-4 1e-4)
DROPOUT=(0.1 0.2 0.3 0.4 0.5)


for i in "${!MODELS[@]}"; do 
    echo "Model: ${MODELS[$i]}\n" >> output.txt
    echo "Batch size: ${BATCH[$i]}\n" >> output.txt
    echo "Weight initialization epochs: ${FEEPOCHS[$i]}\n" >> output.txt
    echo "Weight initialization learning rate: ${FELR[$i]}\n" >> output.txt
    echo "Fine tuning learning rate: ${FTLR[$i]}\n" >> output.txt
    echo "Dropout rate: ${DROPOUT[$i]}\n" >> output.txt

    python3 main.py /home/fmts/msc/experiments/data/isic2019/sampled_unbalanced_5000/ \
        --training \
        --batchsize "${BATCH[i]}" \
        --maxqueuesize 10 \
        --model "${MODELS[i]}" \
        --feepochs "${FEEPOCHS[i]}" \
        --felr "${FELR[i]}" \
        --ftlr "${FTLR[i]}" \
        --dropout "${DROPOUT[i]}"
done

# L2
MODELS=("DenseNet201" "DenseNet201" "DenseNet201" "DenseNet201" "DenseNet201" "DenseNet201")
BATCH=(16 16 16 16 16 16)
FEEPOCHS=(2 2 2 2 2 2)
FELR=(1e-3 1e-3 1e-3 1e-3 1e-3 1e-3)
FTLR=(1e-4 1e-4 1e-4 1e-4 1e-4 1e-4)
L2=(0.1 0.01 0.001 0.0001 0.00001 0.000001)

for i in "${!L2[@]}"; do 
    echo "Model: ${MODELS[$i]}\n" >> output.txt
    echo "Batch size: ${BATCH[$i]}\n" >> output.txt
    echo "Weight initialization epochs: ${FEEPOCHS[$i]}\n" >> output.txt
    echo "Weight initialization learning rate: ${FELR[$i]}\n" >> output.txt
    echo "Fine tuning learning rate: ${FTLR[$i]}\n" >> output.txt
    echo "L2 rate: ${L2[$i]}\n" >> output.txt

    python3 main.py /home/fmts/msc/experiments/data/isic2019/sampled_unbalanced_5000/ \
        --training \
        --batchsize "${BATCH[i]}" \
        --maxqueuesize 10 \
        --model "${MODELS[i]}" \
        --feepochs "${FEEPOCHS[i]}" \
        --felr "${FELR[i]}" \
        --ftlr "${FTLR[i]}" \
        --l2 "${L2[i]}" 
done


# PATIENCE
MODELS=("DenseNet201" "DenseNet201" "DenseNet201" "DenseNet201" "DenseNet201" "DenseNet201")
BATCH=(16 16 16 16 16 16)
FEEPOCHS=(2 2 2 2 2 2)
FELR=(1e-3 1e-3 1e-3 1e-3 1e-3 1e-3)
FTLR=(1e-4 1e-4 1e-4 1e-4 1e-4 1e-4)
PATIENCE=(2 4 8 12 14 16)

for i in "${!PATIENCE[@]}"; do 

    python3 main.py /home/fmts/msc/experiments/data/isic2019/sampled_unbalanced_5000/ \
        --training \
        --batchsize "${BATCH[i]}" \
        --maxqueuesize 10 \
        --model "${MODELS[i]}" \
        --feepochs "${FEEPOCHS[i]}" \
        --felr "${FELR[i]}" \
        --ftlr "${FTLR[i]}" \
        --patience "${PATIENCE[i]}" 
done