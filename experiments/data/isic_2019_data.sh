#! /bin/bash
set -euo pipefail

#rm -rf ./isic2019/ && mkdir -p ./isic2019/

# Download and unzip
#echo "Downloading ISIC 2019 training data..."
#curl -SL https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_Input.zip > ./isic2019/ISIC_2019_Training_Input.zip
#unzip ./isic2019/ISIC2019_Training_Input.zip -d ./isic2019

#echo "Downloading ISIC 2019 ground truth..."
#curl -SL https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_GroundTruth.csv > ./isic2019/ISIC_2019_Training_GroundTruth_Original.csv

#echo "Downloading ISIC 2019 test data..."
#curl -SL https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Test_Input.zip > ./isic2019/ISIC_2019_Test_Input.zip
#unzip ./isic2019/ISIC_2019_Test_Input.zip -d ./isic2019

# Process and compress
for target_size in 224; do
    echo "Processing data for target size "$target_size"x"$target_size"..."
    rm -rf ./isic2019/ISIC_2019_Training_Input/augmented/"$target_size"
    mkdir -p ./isic2019/ISIC_2019_Training_Input/augmented/"$target_size"/{train,test}
    python augment.py --images ./isic2019/ISIC_2019_Training_Input \
                         --descriptions ./isic2019/ISIC_2019_Training_GroundTruth_Original.csv \
                         --target-size "$target_size" \
                         --target-samples 30000 \
                         --output ./isic2019/ISIC_2019_Training_Input/augmented/"$target_size"
done
