import argparse
import itertools
import math
import os
import random
import shutil
import time
from collections import Counter
from math import floor

import numpy as np
import pandas as pd
import PIL
import sklearn.model_selection
import tensorflow as tf
from Augmentor import Operations
from tqdm import tqdm

from data import load_isic_training_data, load_isic_training_and_out_dist_data, train_validation_split, get_dataframe_from_img_folder


# Mean and STD calculated over the Training Set
# Mean:[0.6236094091893962, 0.5198354883713194, 0.5038435406338101]
# STD:[0.2421814437693499, 0.22354427793687906, 0.2314805420919389]
def standardize(
    x, 
    mean=[0.6236, 0.5198, 0.5038], 
    std=[0.2422, 0.2235, 0.2315]
):

    x /= 255.

    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

    x[..., 0] /= std[0]
    x[..., 1] /= std[1]
    x[..., 2] /= std[2]

    return x


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
        """
        arr = np.array(img).astype(float)
        new_img = PIL.Image.fromarray(standardize(arr).astype(np.uint8),'RGB')

        return new_img

    img = PIL.Image.open(filename).convert('RGB')
    img = _crop(img)
    img = _resize(img, target_size)
    #img = _correct(img)
    return img


def augment(source_dataframe, operations, n, img_size):
    n = n - source_dataframe.shape[0] 
    source_dataframe['img'] = source_dataframe.apply(lambda row: load_image(row.path, target_size=img_size), axis=1)

    augmentor_df = source_dataframe.sample(n=n, replace=True)

    with tqdm(total=augmentor_df.shape[0], desc="Executing Pipeline", unit=" Samples") as progress_bar:
        for _, row in augmentor_df.iterrows():

            augmented_image = row["img"]
            for operation in operations:
                r = round(random.uniform(0, 1), 1)
                if r <= operation.probability:
                    augmented_image = operation.perform_operation([augmented_image])[0]

            row['img'] = augmented_image
            row['image'] = f"{row['image']}_{random.randrange(1000)}"

            source_dataframe = source_dataframe.append(row)
            
            progress_bar.set_description("Processing %s" % os.path.basename(row["image"]))
            progress_bar.update(1)

    return source_dataframe


def get_augmentation_operations():
    operations = []
    # Rotate the image by 90 degrees randomly
    operations.append(Operations.Rotate(probability=0.5, rotation=-1))
    # Flip top/bottom
    operations.append(Operations.Flip(probability=0.5, top_bottom_left_right="TOP_BOTTOM"))
    # Flip left/right
    operations.append(Operations.Flip(probability=0.5, top_bottom_left_right="LEFT_RIGHT"))
    # Shear Image
    operations.append(Operations.Shear(probability=0.6, max_shear_left=20, max_shear_right=20))
    # Random change brightness of the image
    operations.append(Operations.RandomBrightness(probability=0.5, min_factor=0.9,max_factor=1.1))
    # Random change saturation of the image
    operations.append(Operations.RandomColor(probability=0.5, min_factor=0.9,max_factor=1.1))

    return operations


def sample(df_ground_truth, images_path, count_per_category, img_size=(224,224)):
    result = pd.DataFrame()
    for i, _ in enumerate(count_per_category):
        category_samples = df_ground_truth.loc[df_ground_truth['category'] == i]
        if(count_per_category[i] > category_samples.shape[0]):
            # oversample
            print(f'Augmenting {category_samples.shape[0]} samples from class {i} into approximately {count_per_category[i]} samples...')
            samples = augment(category_samples, get_augmentation_operations(), count_per_category[i], img_size)
        elif(count_per_category[i] < category_samples.shape[0]):
            # keep undersample 
            print(f'Undersampling {category_samples.shape[0]} samples from class {i} into approximately {count_per_category[i]} samples...')
            samples = category_samples.sample(n=count_per_category[i])
            samples['img'] = samples.apply(lambda row: load_image(os.path.join(images_path, row.image+'.jpg'), target_size=img_size), axis=1)
        else:
            # Keep original samples
            print(f'Keeping original {category_samples.shape[0]} samples from class {i} ...')
            samples = category_samples.copy()
            samples['img'] = category_samples.apply(lambda row: load_image(os.path.join(images_path, row.image+'.jpg'), target_size=img_size), axis=1)
        
        result = result.append(samples) 

    # Shuffle rows and return result
    return result.sample(frac=1).reset_index(drop=True) 


def process(
    images_path, 
    test_folder,
    descriptions_filename, 
    target_img_size, 
    training_samples, 
    class_balance,
    unknown_images_path=None,
    unknown_train=False
):

    if unknown_images_path and unknown_train is True:
        df_ground_truth, category_names = load_isic_training_and_out_dist_data(images_path, descriptions_filename, unknown_images_path)
    else:
        df_ground_truth, category_names, unknown_category = load_isic_training_data(images_path, descriptions_filename)

    if test_folder is None:
        df_train, df_test = train_validation_split(df_ground_truth)
    else:
        df_train = df_ground_truth
        df_test = get_dataframe_from_img_folder(test_folder)

    if unknown_images_path and unknown_train is not True:
        df_out_dist = get_dataframe_from_img_folder(unknown_images_path, has_path_col=True)
        for name in category_names:
            df_out_dist[name] = 0.0
        df_out_dist[unknown_category] = 1.0
        df_out_dist["category"] = 8
        # Change the order of columns
        df_out_dist = df_out_dist[df_train.columns.values]
        df_test = df_test.append(df_out_dist)


    # Process train set + oversample/undersample class samples
    count_per_category = Counter(df_train['category'])
    total_sample_count = sum(count_per_category.values())

    # Calculate number of samples per class
    samples_per_category = [None] * len(count_per_category)
    if class_balance:
        samples_per_category = [floor(training_samples/len(count_per_category)) for i in count_per_category]
    else:
        for i, _ in count_per_category.most_common():
            if (training_samples is None):
                # Keep original
                samples_per_category[i] = count_per_category[i]
            else:
                # Oversample/undersample
                count_ratio = float(count_per_category[i])/total_sample_count
                samples_per_category[i] = floor(count_ratio*training_samples)
        
        if (training_samples is not None):
            # Adjust to fill remainder samples due to flooring calculations
            samples_per_category[-1] = samples_per_category[-1] + (training_samples - sum(samples_per_category)) 
    
    print(f'Turning {total_sample_count} samples into {sum(samples_per_category)} samples...')
    result = sample(
        df_train,
        images_path, 
        samples_per_category,
        img_size = target_img_size
    )

    # Process test set
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    df_test['img'] = df_test.apply(lambda row: load_image(row.path, target_size=target_img_size), axis=1)

    # Removing unnecessary path column
    del result['path']
    del df_test['path']

    return result, df_test


def save(df, images_path, descriptions_path):
    if os.path.exists(images_path):
        shutil.rmtree(images_path)

    os.makedirs(images_path)
    if os.path.exists(descriptions_path):
        os.remove(descriptions_path)

    for _, row in df.iterrows():
        row["img"].save(os.path.join(images_path, f'{row["image"]}.jpg'))
    
    del df['img']
    del df['category']

    df.to_csv(descriptions_path, index=False)

def save_no_unknown(df_test, no_unknown_descriptions_file):
    df_test_no_unknown = df_test.copy()
    del df_test_no_unknown['img']
    df_test_no_unknown = df_test_no_unknown[df_test_no_unknown.category != 8]
    del df_test_no_unknown['category']
    df_test_no_unknown.to_csv(no_unknown_descriptions_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default="./isic2019/ISIC_2019_Training_Input")
    parser.add_argument('--test', default=None)
    parser.add_argument('--unknown-images', default=None)
    parser.add_argument('--unknown-train', dest='unknown_train', action='store_true', default=False)
    parser.add_argument('--descriptions', default="./isic2019/ISIC_2019_Training_GroundTruth.csv")
    parser.add_argument('--target-size', type=int, default=224)
    parser.add_argument('--training-samples', type=int, default=None)
    parser.add_argument('--class-balance', dest='classbalance', action='store_true', default=False)
    parser.add_argument('--output', default="./isic2019/sampled", required=True)
    args = parser.parse_args()

    if(args.classbalance and not args.training_samples):
        raise ValueError("Must indicate --training-samples when using --class-balance") 

    df_train, df_test = process(
        args.images, 
        args.test,
        args.descriptions, 
        (args.target_size, args.target_size), 
        args.training_samples, 
        args.classbalance,
        unknown_images_path=args.unknown_images,
        unknown_train=args.unknown_train
    )
    
    save(
        df_train, 
        os.path.join(args.output, 'ISIC_2019_Training_Input'), 
        os.path.join(args.output, 'ISIC_2019_Training_GroundTruth.csv')
    )

    if args.unknown_images is not None:
        # Save test ground truth file without unknown samples
        save_no_unknown(
            df_test, 
            os.path.join(args.output, 'ISIC_2019_Test_GroundTruth.csv')
        )

    save(
        df_test, 
        os.path.join(args.output, 'ISIC_2019_Test_Input'), 
        os.path.join(
            args.output, 
            'ISIC_2019_Test_GroundTruth_Unknown.csv' if args.unknown_images is not None else 'ISIC_2019_Test_GroundTruth.csv' 
        )
    )
    
