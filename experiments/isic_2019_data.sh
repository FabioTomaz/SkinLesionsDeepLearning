#! /bin/bash
set -euo pipefail

rm -rf ./data/isic2019/ && mkdir -p ./data/isic2019/

# Download and unzip
echo "Downloading ISIC 2019 training data..."
curl -SL https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_Input.zip > ./data/isic2019/ISIC_2019_Training_Input.zip
unzip ./data/isic2019/ISIC2019_Training_Input.zip -d ./data/isic2019

echo "Downloading ISIC 2019 ground truth..."
curl -SL https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_GroundTruth.csv > ./data/isic2019/ISIC_h2019_Training_GroundTruth.csv

echo "Downloading ISIC 2019 test data..."
curl -SL https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Test_Input.zip > ./data/isic2019/ISIC_2019_Test_Input.zip
unzip ./data/isic2019/ISIC_2019_Test_Input.zip -d ./data/isic2019
