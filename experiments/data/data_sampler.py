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
from Augmentor import DataFramePipeline, Pipeline
from Augmentor.ImageUtilities import AugmentorImage
from tqdm import tqdm

from data import train_validation_split


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


def load_image(filename, target_size=(500,500)):
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
    #img = _crop(img)
    #img = _resize(img, target_size)
    #img = _correct(img)
    return img


def scan_dataframe(source_dataframe, image_col, category_col, output_directory):
    # ensure column is categorical
    cat_col_series = pd.Categorical(source_dataframe[category_col])
    abs_output_directory = os.path.abspath(output_directory)
    class_labels = list(enumerate(cat_col_series.categories))

    augmentor_images = []

    for image_path, cat_id in zip(source_dataframe[image_col].values, cat_col_series.codes):
        a = AugmentorImage(image_path=image_path, output_directory=abs_output_directory)
        a.class_label = str(cat_id)
        a.class_label_int = cat_id
        categorical_label = np.zeros(len(class_labels), dtype=np.uint32)
        categorical_label[cat_id] = 1
        a.categorical_label = categorical_label
        a.file_format = os.path.splitext(image_path)[1].split(".")[1]
        augmentor_images.append(a)

    return augmentor_images, class_labels


class CustomDataFramePipeline(Pipeline):
    def __init__(self, source_dataframe, image_col, category_col, output_directory="output", save_format="JPEG"):
        super(CustomDataFramePipeline, self).__init__(source_directory=None,
                                                output_directory=output_directory,
                                                save_format=save_format)
        self._source_dataframe = source_dataframe
        self._populate(source_dataframe, image_col, category_col, output_directory, save_format)

    def _populate(self, source_dataframe, image_col, category_col, output_directory, save_format):
        self.augmentor_images, self.class_labels = scan_dataframe(source_dataframe, image_col, category_col, output_directory)
        self._check_images(output_directory)

    def sample(self, images_path, n):
        n = n - self._source_dataframe.shape[0] 
        self._source_dataframe['img'] = self._source_dataframe.apply(lambda row: load_image(os.path.join(images_path, row.image+'.jpg')), axis=1)

        augmentor_df = self._source_dataframe.sample(n=n, replace=True)

        with tqdm(total=augmentor_df.shape[0], desc="Executing Pipeline", unit=" Samples") as progress_bar:
            for index, row in augmentor_df.iterrows():

                augmented_image = row["img"]
                for operation in self.operations:
                    r = round(random.uniform(0, 1), 1)
                    if r <= operation.probability:
                        augmented_image = operation.perform_operation([augmented_image])[0]

                row['img'] = augmented_image
                row['image'] = f"{row['image']}_{index}"

                self._source_dataframe = self._source_dataframe.append(row)
                
                progress_bar.set_description("Processing %s" % os.path.basename(row["image"]))
                progress_bar.update(1)

        return self._source_dataframe


def sample(df_ground_truth, images_path, count_per_category, img_size=None):
    result = pd.DataFrame()
    for i, _ in enumerate(count_per_category):
        category_samples = df_ground_truth.loc[df_ground_truth['category'] == i]
        if(count_per_category[i] > category_samples.shape[0]):
            # oversample
            n_augment = count_per_category[i] - category_samples.shape[0]
            print(f'Augmenting {category_samples.shape[0]} samples from class {i} into approximately {count_per_category[i]} samples...')
            pipeline = get_augmentation_pipeline(category_samples, "test_smaples")
            samples = pipeline.sample(images_path, count_per_category[i])
        else:
            # undersample
            print(f'Undersampling {category_samples.shape[0]} samples from class {i} into approximately {count_per_category[i]} samples...')
            samples = category_samples.sample(n=count_per_category[i])
            samples['img'] = samples.apply(
                lambda row: load_image(os.path.join(images_path, row.image+'.jpg')), 
                axis=1
            )
        result = result.append(samples) 

    return result.sample(frac=1).reset_index(drop=True) 


def get_augmentation_pipeline(df, output):
    p_train = CustomDataFramePipeline(
        df, 
        "path",
        "category",
        output_directory=output
    )

    # Rotate the image by either 90, 180, or 270 degrees randomly
    p_train.rotate_random_90(probability=0.5)
    # Flip the image along its vertical axis
    p_train.flip_top_bottom(probability=0.5)
    # Flip the image along its horizontal axis
    p_train.flip_left_right(probability=0.5)
    # Shear image
    p_train.shear(probability=0.5, max_shear_left=20, max_shear_right=20)
    # Random change brightness of the image
    p_train.random_brightness(probability=0.5, min_factor=0.9, max_factor=1.1)
    # Random change saturation of the image
    p_train.random_color(probability=0.5, min_factor=0.9, max_factor=1.1)

    return p_train

def process(images_path, descriptions_filename, target_img_size, target_m, class_balance):

    # Ground truth
    df_ground_truth, known_category_names, _ = load_isic_training_data(images_path, descriptions_filename)    
    count_per_category = Counter(df_ground_truth['category'])
    total_sample_count = sum(count_per_category.values())

    if (target_m == None or target_m == total_sample_count):
        result = df_ground_truth.sample(frac=1).reset_index(drop=True)
        result['img'] = result.apply(lambda row: load_image(os.path.join(images_path, row.image+'.jpg')), axis=1)
    else:
        samples_per_category = []

        if class_balance:
            samples_per_category = [floor(target_m/len(count_per_category)) for i in count_per_category]
        else:
            for i, _ in count_per_category.most_common():
                count_ratio = float(count_per_category[i])/total_sample_count
                samples_per_category.insert(i, floor(count_ratio*target_m))
        
        print(f'Turning {total_sample_count} samples into approximately {target_m} samples...')
        result = sample(
            df_ground_truth,
            images_path, 
            samples_per_category,
            img_size = target_img_size
        )

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
    parser.add_argument('--unknown-images', default=None)
    parser.add_argument('--descriptions', default="./isic2019/ISIC_2019_Training_GroundTruth_Original.csv")
    parser.add_argument('--target-size', type=int, default=224)
    parser.add_argument('--target-samples', type=int, default=None)
    parser.add_argument('--class-balance', dest='classbalance', action='store_true', default=False)
    parser.add_argument('--output', default="./isic2019/sampled", required=True)
    args = parser.parse_args()

    train_output_path = os.path.join(args.output, 'ISIC_2019_Training_Input') 
    test_output_path = os.path.join(args.output, 'ISIC_2019_Test_Input')
    unknown_output_path = os.path.join(args.output, 'Unknown')

    if not os.path.exists(train_output_path):
        os.makedirs(train_output_path)
    if not os.path.exists(test_output_path):
        os.makedirs(test_output_path)
    if not os.path.exists(unknown_output_path):
        os.makedirs(unknown_output_path)

    resultDf = process(
        args.images, 
        args.descriptions, 
        (args.target_size, args.target_size), 
        args.target_samples, 
        args.classbalance
    )
    del resultDf['path']
    
    df_train, df_test = train_validation_split(resultDf)

    save(df_train, train_output_path, os.path.join(args.output, 'ISIC_2019_Training_GroundTruth.csv'))
    save(df_test, test_output_path, os.path.join(args.output, 'ISIC_2019_Test_GroundTruth.csv'))
