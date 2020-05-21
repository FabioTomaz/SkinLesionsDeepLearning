# Machine Learning for Automated Diagnosis of Skin Lesions

The MSc thesis by Fábio Santos who studied multiple approaches for multi-class classification of skin lesions using the ISIC 2019 dataset. Some of the main topics include pre-trained model study, hyperparameter optimization, data augmentation study, ensemble study and out of distribution study.

## Environment

We use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage a virtual Python 3.6 environment and dependencies.

Install Miniconda and create the virtual Python environment with all necessary dependencies.

Activate and enter the environment at any point with:

```
conda activate experiments
```

## Data

Download the official ISIC2019 train set using the following:

```
./experiments/data/isic_2019_data.sh
```

## Preprocess, augment and split data into train, test and validation sets

Example:
```
python3 ./experiments/data/data_sampler_copy.py \           
        --output "./experiments/data/isic2019/sampled_balanced_60000" \
        --images "./experiments/data/isic2019/ISIC_2019_Training_Input" \ 
        --unknown-images "./experiments/data/isic_archive/unknown" \
        --descriptions "./experiments/data/isic2019/ISIC_2019_Training_GroundTruth.csv" \
        --target-size 224 \
        --data-augmentation-group 1 \
        --class-balance \
        --training-samples 60000 
        --unknown-train 
```

## Train

Run the prepared scripts to train the VGG16 transfer learning models and the custom CNN end-to-end learning models:

```
python3 main.py ./experiments/data/isic2019/sampled_balanced_60000/ \
                --training \
                --modelfolder models \
                --historyfolder history \
                --unknown 1 \
                --batchsize 32 \
                --maxqueuesize 10 \
                --model DenseNet121 \
                --feepochs 2 \
                --ftepochs 100 \
                --felr 1e-3 \
                --ftlr 1e-4 \
                --online-data-augmentation-group 1
```

## Test

Run the prepared scripts to test the VGG16 transfer learning models and the custom CNN end-to-end learning models:

```
python3 main.py ./experiments/data/isic2019/sampled_balanced_60000/ \
                --predtest --predtestresultfolder test_predict_results \
                --modelfolder models \
                --historyfolder history \
                --unknown 1 \
                --batchsize 32 \
                --maxqueuesize 10 \
                --model DenseNet121 \
                --feepochs 2 \
                --ftepochs 100 \
                --felr 1e-3 \
                --ftlr 1e-4 \
                --online-data-augmentation-group 1
```

## Results
You can see the results of this dissertation in the notebooks of folder "./experiments/notebooks". Additionally, all dissertation images are also placed in this folder (in pdf format).

In order to replicate the results one must simply re-run the notebooks. 

## Dissertation

The LaTeX document in `doc/` is the dissertation for the MSc thesis which tells the story of the experiments.

```
cd doc
make
```
