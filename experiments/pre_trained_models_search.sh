#! /bin/bash
set -euxo pipefail

echo "===================== Only train classifer for all models ===================== \n" 

for model in ResNet50 ResNet101 ResNet152 DenseNet121 DenseNet169 DenseNet201 Xception InceptionV3 InceptionResNetV2 EfficientNetB0 EfficientNetB1 EfficientNetB2; do

    echo "Model: ${model}\n" 

    python3 main.py /home/fmts/msc/experiments/data/isic2019/sampled_unbalanced_5000 \
    --training --predtest --predtestresultfolder test_predict_results \
    --batchsize 16 \
    --maxqueuesize 10 \
    --model "$model" \
    --feepochs 100 \
    --ftepochs 0 \
    --felr 1e-4 \
    --ftlr 1e-4
done


# echo "===================== Fine tune all models ===================== \n" 

# for model in VGG16 VGG19 ResNet50 ResNet101 ResNet152 DenseNet121 DenseNet169 DenseNet201 Xception InceptionV3 InceptionResNetV2 EfficientNetB0 EfficientNetB1 EfficientNetB2; do

#     echo "Model: ${model}\n" 

#     python3 main.py /home/fmts/msc/experiments/data/isic2019/sampled_unbalanced_5000 \
#     --training --predtest --predtestresultfolder test_predict_results \
#     --batchsize 16 \
#     --maxqueuesize 10 \
#     --model "$model" \
#     --feepochs 0 \
#     --ftepochs 100 \
#     --felr 1e-4 \
#     --ftlr 1e-4
# done