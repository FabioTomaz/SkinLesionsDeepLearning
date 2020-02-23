import os
import argparse
import itertools
import math

import sklearn.model_selection
import tensorflow as tf
import numpy as np
import PIL

import os
import shutil
import time
from math import floor

import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from collections import Counter

from data import load_isic_training_data, train_validation_split, compute_class_weight_dict, get_dataframe_from_img_folder

AUGMENTATIONS = [
    lambda x: x.transpose(PIL.Image.FLIP_LEFT_RIGHT),
    lambda x: x.transpose(PIL.Image.FLIP_TOP_BOTTOM),
    lambda x: x.transpose(PIL.Image.ROTATE_90),
    lambda x: x.transpose(PIL.Image.ROTATE_180),
    lambda x: x.transpose(PIL.Image.ROTATE_270),
]

AUGMENTATIONS = [c for j in range(1, len(AUGMENTATIONS)+1) for c in itertools.combinations(AUGMENTATIONS, j)]


def augment(img, target_size, i):
    for f in AUGMENTATIONS[i]:
        img = f(load_image(img))
    return img

def standardize(x, mean, std):
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

    x[..., 0] /= std[0]
    x[..., 1] /= std[1]
    x[..., 2] /= std[2]

    return x

def is_dir(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def is_file(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)

def load_image(filename, target_size=(224,224)):
    assert target_size[0] == target_size[1]

    def _crop(img):
        width, height = img.size
        if width == height:
            return img

        length = min(width, height)

        left = (width - length) // 2
        upper = (height - length) // 2
        right = left + length
        lower = upper + length

        box = (left, upper, right, lower)
        return img.crop(box)

    def _resize(img, target_size):
        return img.resize(target_size, PIL.Image.NEAREST)

    def _correct(img):
        """
        Normalize PIL image

        Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
        """
        img_y, img_b, img_r = img.convert('YCbCr').split()

        img_y_np = np.asarray(img_y).astype(float)

        img_y_np /= 255
        img_y_np -= img_y_np.mean()
        img_y_np /= img_y_np.std()
        scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                        np.abs(np.percentile(img_y_np, 99.0))])
        img_y_np = img_y_np / scale
        img_y_np = np.clip(img_y_np, -1.0, 1.0)
        img_y_np = (img_y_np + 1.0) / 2.0

        img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

        img_y = PIL.Image.fromarray(img_y_np)

        img_ybr = PIL.Image.merge('YCbCr', (img_y, img_b, img_r))
        return img_ybr.convert('RGB')

    img = PIL.Image.open(filename).convert('RGB')
    #img = _crop(img)
    #img = _resize(img, target_size)
    #img = _correct(img)
    return img

def undersample(df_ground_truth, count_per_category):
    result = pd.DataFrame()
    for i, _ in enumerate(count_per_category):
        category_samples = df_ground_truth.loc[df_ground_truth['category'] == i]
        result = result.append(category_samples.sample(n=count_per_category[i])) 
    result = result.sample(frac=1).reset_index(drop=True)
    return result

def process(images_path, descriptions_filename, target_img_size, target_m, class_balance):

    # Ground truth
    df_ground_truth, known_category_names, _ = load_isic_training_data(images_path, descriptions_filename)    
    count_per_category = Counter(df_ground_truth['category'])
    total_sample_count = sum(count_per_category.values())
    count_per_category_ratio = []
    samples_per_category = []

    if (target_m == None or target_m == total_sample_count):
        result = df_ground_truth.sample(frac=1).reset_index(drop=True)
        result['img'] = result.apply(lambda row: load_image(os.path.join(images_path, row.image+'.jpg')), axis=1)
    elif target_m > total_sample_count:
        # Augment minority until classes are balanced and augmentation goal is reached
        print(f'Augmenting {len(y)} samples to approximately {target_m} class-balanced samples...')

        if class_balance:
            class_weight_dict, _ = compute_class_weight_dict(df_ground_truth)

            # Group data into classes for class balancing
            class_samples_n = [(xv, yv) for (xv,yv) in zip(x,y) if yv == class_i for class_i in range(0,8)]

            for S in class_samples_n:
                augmentation_factor = (target_m//8) // len(S)
                assert len(AUGMENTATIONS) >= augmentation_factor
                S += [(augment(xv, target_img_size, j), yv) for (xv,yv) in S for j in range(augmentation_factor)]

            # Construct NumPy ndarrays out of the lists
            total = minority + majority
            x = np.array([np.asarray(xv, dtype='float32') for (xv,yv) in total], dtype='float32')
            y = np.array([yv for (xv,yv) in total], dtype='float32')
   
            # Ensure the preprocessed dataset's classes were reasonably balanced
            _, counts = np.unique(y, return_counts=True)
            assert abs(counts[0] - counts[1]) < 100
        else:
            print("Not implemented yet")
    else:
        for i, _ in count_per_category.most_common():
            count_ratio = float(count_per_category[i])/total_sample_count
            count_per_category_ratio.insert(i, count_ratio)
            samples_per_category.insert(i, floor(count_ratio*target_m))
        
        if class_balance == True:
            print(f'Undersampling {total_sample_count} samples to approximately {target_m} class-balanced samples...')
            min_samples = min(samples_per_category)
            target_samples = [min_samples for i in samples_per_category]
            result = undersample(df_ground_truth, target_samples)
        else:
            print(f'Undersampling {total_sample_count} samples to approximately {target_m} class-inbalanced samples...')
            result = undersample(df_ground_truth, samples_per_category)
        result['img'] = result.apply(lambda row: load_image(os.path.join(images_path, row.image+'.jpg')), axis=1)

    return result


def load(preprocessed_dataset_filename):
    dataset = np.load(preprocessed_dataset_filename)
    return dataset['x'], dataset['y']


def save(df, images_path, descriptions_path):
    for index, row in df.iterrows():
        row["img"].save(os.path.join(images_path, f'{row["image"]}.jpg'))
    
    del df['img']
    del df['category']

    df.to_csv(descriptions_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default="./isic2019/ISIC_2019_Training_Input")
    parser.add_argument('--descriptions', default="./isic2019/ISIC_2019_Training_GroundTruth_Original.csv")
    parser.add_argument('--target-size', type=int, default=224)
    parser.add_argument('--target-samples', type=int, default=None)
    parser.add_argument('--class-balance', type=bool, default=False)
    parser.add_argument('--output', default="./isic2019/sampled", required=True)
    args = parser.parse_args()

    train_output_path = os.path.join(args.output, 'ISIC_2019_Training_Input') 
    test_output_path = os.path.join(args.output, 'ISIC_2019_Test_Input')

    if not os.path.exists(train_output_path):
        os.makedirs(train_output_path)
    if not os.path.exists(test_output_path):
        os.makedirs(test_output_path)

    resultDf = process(
        args.images, 
        args.descriptions, 
        (args.target_size, args.target_size), 
        args.target_samples, 
        args.class_balance
    )
    del resultDf['path']
    
    df_train, df_test = train_validation_split(resultDf)

    save(df_train, train_output_path, os.path.join(args.output, 'ISIC_2019_Training_GroundTruth.csv'))
    save(df_test, test_output_path, os.path.join(args.output, 'ISIC_2019_Test_GroundTruth.csv'))