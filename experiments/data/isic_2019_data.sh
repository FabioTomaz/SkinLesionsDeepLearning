#! /bin/bash
set -euo pipefail

rm -rf ./isic2019/ && mkdir -p ./isic2019/

# Download and unzip
echo "Downloading ISIC 2019 training data..."
curl -SL https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_Input.zip > ./isic2019/ISIC_2019_Training_Input.zip
unzip ./isic2019/ISIC2019_Training_Input.zip -d ./isic2019

echo "Downloading ISIC 2019 ground truth..."
curl -SL https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_GroundTruth.csv > ./isic2019/ISIC_2019_Training_GroundTruth_Original.csv

echo "Downloading ISIC 2019 test data..."
curl -SL https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Test_Input.zip > ./isic2019/ISIC_2019_Test_Input.zip
unzip ./isic2019/ISIC_2019_Test_Input.zip -d ./isic2019

echo "Download unknown examples..."
curl -o ./isic_archive/Unknown.zip -SL "https://isic-archive.com/api/v1/image/download?include=all&imageIds=%5B%2254e755f8bae47850e86ce012%22,%2254e755fdbae47850e86ce058%22,%22558d6091bae47801cf7343bd%22,%22558d60b2bae47801cf73446e%22,%22558d60bebae47801cf7344b6%22,%22558d60bfbae47801cf7344bf%22,%22558d60c8bae47801cf7344f5%22,%22558d60ccbae47801cf73450d%22,%22558d60d5bae47801cf73453a%22,%22558d60d7bae47801cf73454c%22,%22558d60d9bae47801cf734552%22,%22558d60ddbae47801cf734570%22,%22558d60dfbae47801cf734579%22,%22558d60e1bae47801cf734585%22,%22558d60e7bae47801cf7345ac%22,%22558d6124bae47801cf73470e%22,%22558d6126bae47801cf734714%22,%22558d612dbae47801cf73473e%22,%22558d612ebae47801cf734744%22,%22558d612fbae47801cf73474a%22,%22558d6134bae47801cf734762%22,%22558d6134bae47801cf734765%22,%22558d613bbae47801cf73478c%22,%22558d6140bae47801cf7347a7%22,%22558d6164bae47801cf734876%22,%22558d616fbae47801cf7348b2%22,%22558d6170bae47801cf7348b5%22,%22558d617fbae47801cf73490f%22,%22558d6187bae47801cf73494b%22,%22558d6192bae47801cf73497e%22,%22558d6198bae47801cf734987%22,%22558d61a1bae47801cf7349ba%22,%22558d61a1bae47801cf7349bd%22,%22558d62d9bae47801cf7349e7%22,%22558d62dcbae47801cf7349f9%22,%22558d62e0bae47801cf734a0e%22,%22558d62e7bae47801cf734a38%22,%22558d62f4bae47801cf734a86%22,%22558d62f6bae47801cf734a98%22,%22558d62fabae47801cf734aaa%22,%22558d6302bae47801cf734ada%22,%22558d6311bae47801cf734b34%22,%22558d6311bae47801cf734b37%22,%22558d6318bae47801cf734b5e%22,%22558d6318bae47801cf734b61%22,%22558d6319bae47801cf734b6a%22,%22558d633cbae47801cf734c3c%22,%22558d6343bae47801cf734c66%22,%22558d6345bae47801cf734c72%22,%22558d634dbae47801cf734c9c%22,%22558d6361bae47801cf734d0e%22,%22558d6385bae47801cf734df8%22,%22558d638fbae47801cf734e2b%22,%22558d6393bae47801cf734e43%22,%22558d639cbae47801cf734e73%22,%22558d63a7bae47801cf734eb5%22,%22558d63abbae47801cf734ed0%22,%22558d63acbae47801cf734ed3%22,%22558d63c4bae47801cf734f66%22,%22558d63cabae47801cf734f8a%22,%22558d63cdbae47801cf734f99%22,%22558d63d9bae47801cf734fe1%22,%22558d63f2bae47801cf735074%22,%22558d63f3bae47801cf735077%22,%22558d6408bae47801cf7350f8%22,%22558d640abae47801cf735104%22,%22558d640cbae47801cf735110%22,%22558d6426bae47801cf7351a6%22,%22558d6427bae47801cf7351af%22,%22558d6431bae47801cf7351e8%22,%22558d6432bae47801cf7351eb%22,%22558d6432bae47801cf7351ee%22,%22558d6432bae47801cf7351f1%22,%22558d6437bae47801cf735209%22,%22558d6444bae47801cf73525a%22,%22558d6448bae47801cf735272%22,%22558d6454bae47801cf7352c0%22,%22558d6467bae47801cf73532c%22,%22558d6477bae47801cf735389%22,%22558d6478bae47801cf73538c%22%5D"
unzip ./isic_archive/Unknown.zip -d ./isic_archive/unknown

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
