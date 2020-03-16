#! /bin/bash
set -euo pipefail

# Download and unzip
rm -rf ./isic2018/ && mkdir -p ./isic2018/

echo "Downloading ISIC 2019 training data..."
curl -sSL https://challenge.kitware.com/api/v1/item/5ac20fc456357d4ff856e139/download > ./isic2018/ISIC2018_Task3_Training_Input.zip

echo "Downloading ISIC 2018 ground truth..."
curl -sSL https://challenge.kitware.com/api/v1/item/5ac20eeb56357d4ff856e136/download > ./isic2018/ISIC2018_Task3_Training_GroundTruth.zip

unzip ./isic2018/ISIC2018_Task3_Training_Input.zip -d ./isic2018
unzip ./isic2018/ISIC2018_Task3_Training_GroundTruth.zip -d ./isic2018

mv ./isic2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv ./isic2018/ISIC2018_Task3_Training_GroundTruth.csv
rm -rf ./isic2018/ISIC2018_Task3_Training_GroundTruth

# Process and compress
for target_size in 224; do
    echo "Processing data for target size "$target_size"x"$target_size"..."
    rm -rf ./isic2018/"$target_size"
    mkdir -p ./isic2018/"$target_size"/{train,test}
    python ./src/data.py --images ./isic2018/ISIC2018_Task3_Training_Input \
                         --descriptions ./isic2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv \
                         --target-size "$target_size" \
                         --target-samples 16000 \
                         --output ./isic2018/"$target_size"
done
