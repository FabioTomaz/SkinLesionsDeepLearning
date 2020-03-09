import math
import os
import pandas as pd
import numpy as np
from Augmentor import Pipeline
from Augmentor.Operations import CropPercentage
from image_iterator import ImageIterator
from keras_numpy_backend import softmax
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from tensorflow.keras.models import Model
import random
import datetime

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class CustomPipeline(Pipeline):
    def perform_operations(self, image):
        augmented_image = image
        for operation in self.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                augmented_image = operation.perform_operation([augmented_image])[0]
        return augmented_image

class LesionClassifier():
    """Base class of skin lesion classifier.
    # Arguments
        batch_size: Integer, size of a batch.
        image_data_format: String, either 'channels_first' or 'channels_last'.
    """
    def __init__(
        self, 
        model_folder, 
        input_size, 
        image_data_format=None, 
        batch_size=32, 
        max_queue_size=10, 
        rescale=None, 
        preprocessing_func=None, 
        class_weight=None,
        num_classes=None, 
        image_paths_train=None, 
        categories_train=None, 
        image_paths_val=None, 
        categories_val=None
    ):

        self.history_folder = 'history'
        self.model_folder = model_folder
        self.input_size = input_size
        if image_data_format is None:
            self.image_data_format = K.image_data_format()
        else:
            self.image_data_format = image_data_format
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.rescale = rescale
        self.preprocessing_func = preprocessing_func
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.image_paths_train = image_paths_train
        self.categories_train = categories_train
        self.image_paths_val = image_paths_val
        self.categories_val = categories_val
        
        self.log_date = datetime.datetime.now().isoformat()

        self.aug_pipeline_train = LesionClassifier.create_aug_pipeline_train(self.input_size)
        print('Image Augmentation Pipeline for Training Set')
        self.aug_pipeline_train.status()

        self.aug_pipeline_val = LesionClassifier.create_aug_pipeline_val(self.input_size)
        print('Image Augmentation Pipeline for Validation Set')
        self.aug_pipeline_val.status()

        self.generator_train, self.generator_val = self._create_image_generator()

    @staticmethod
    def create_aug_pipeline_train(input_size):
        """Image Augmentation Pipeline for Training Set."""

        p_train = CustomPipeline()
        # Random crop
        #p_train.add_operation(CropPercentage(
        #    probability=1, 
        #    percentage_area=0.8, 
        #    centre=False,
        #    randomise_percentage_area=True
        #))
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
        # Resize the image to the desired input size of the model
        # p_train.resize(probability=1, width=input_size[0], height=input_size[1])

        return p_train

    @staticmethod
    def create_aug_pipeline_val(input_size):
        """Image Augmentation Pipeline for Validation/Test Set."""
        p_val = CustomPipeline()
        # # Center Crop
        # p_val.crop_centre(probability=1, percentage_area=0.9)
        # Resize the image to the desired input size of the model
        p_val.resize(probability=1, width=input_size[0], height=input_size[1])
        return p_val

    @staticmethod
    def predict_dataframe(
        model, 
        df, 
        x_col='path', 
        y_col='category', 
        id_col='image', 
        category_names=None,
        augmentation_pipeline=None, 
        preprocessing_function=None,
        batch_size=32, 
        workers=1, 
        unknown_category=None,
        unknown_thresholds=[]
    ):
        unknown_thresholds = [1.0] + unknown_thresholds
        
        generator = ImageIterator(
            image_paths=df[x_col].tolist(),
            labels=None,
            augmentation_pipeline=augmentation_pipeline,
            batch_size=batch_size,
            shuffle=False,  # shuffle must be False otherwise will get a wrong balanced accuracy
            rescale=None,
            preprocessing_function=preprocessing_function,
            pregen_augmented_images=False,  # Only 1 epoch.
            data_format=K.image_data_format()
        )

        # Predict
        # https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer('dense_pred').output)
        logits = intermediate_layer_model.predict_generator(
            generator, 
            verbose=1, 
            workers=workers
        )

        df_softmax_dict = {}

        # convert softmax values to floating point to become valid
        original_softmax_probs = softmax(logits).astype(float) 
        
        # Apply softmax threshold to determine unknown class
        for unknown_thresh in unknown_thresholds:
            softmax_probs = original_softmax_probs.copy()

            if unknown_thresh != 1.0:
                unknown_softmax_values = np.zeros((len(softmax_probs),1))
                for i in range(len(softmax_probs)):
                    if max(softmax_probs[i]) < unknown_thresh:
                        for j in range(len(softmax_probs[i])):
                            softmax_probs[i][j] = 0.0
                        unknown_softmax_values[i] =  1.0
                softmax_probs = np.append(softmax_probs, unknown_softmax_values, axis=1)

            # softmax probabilities
            df_softmax = pd.DataFrame(
                softmax_probs, 
                columns=category_names + [unknown_category] if unknown_thresh != 1.0 else category_names
            )
            if y_col in df.columns:
                df_softmax[y_col] = df[y_col].to_numpy()
            
            df_softmax['pred_'+y_col] = np.argmax(softmax_probs, axis=1)
            df_softmax.insert(0, id_col, df[id_col].to_numpy())

            df_softmax_dict[unknown_thresh] = df_softmax

        return df_softmax_dict

    def _create_image_generator(self):
        ### Training Image Generator
        generator_train = ImageIterator(
            image_paths=self.image_paths_train,
            labels=self.categories_train,
            augmentation_pipeline=self.aug_pipeline_train,
            batch_size=self.batch_size,
            shuffle=True,
            rescale=self.rescale,
            preprocessing_function=self.preprocessing_func,
            pregen_augmented_images=False,
            data_format=self.image_data_format
        )

        ### Validation Image Generator
        generator_val = ImageIterator(
            image_paths=self.image_paths_val,
            labels=self.categories_val,
            augmentation_pipeline=self.aug_pipeline_val,
            batch_size=self.batch_size,
            shuffle=True,
            rescale=self.rescale,
            preprocessing_function=self.preprocessing_func,
            pregen_augmented_images=True, # Since there is no randomness in the augmentation pipeline.
            data_format=self.image_data_format
        )

        return generator_train, generator_val

    def _create_checkpoint_callbacks(self, subdir):
        """Create the functions to be applied at given stages of the training procedure."""

        model_path = os.path.join(self.model_folder, self.model_name, subdir)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        checkpoint_balanced_acc = ModelCheckpoint(
            filepath=os.path.join(model_path, "best_balanced_acc.hdf5"),
            monitor='val_balanced_accuracy',
            verbose=1,
            save_best_only=True
        )

        checkpoint_balanced_acc_weights = ModelCheckpoint(
            filepath=os.path.join(model_path, "best_balanced_acc_weights.hdf5"),
            monitor='val_balanced_accuracy',
            verbose=1,
            save_weights_only=True,
            save_best_only=True
        )
        
        checkpoint_latest = ModelCheckpoint(
            filepath=os.path.join(model_path,  "latest.hdf5"),
            verbose=1,
            save_best_only=False
        )

        checkpoint_loss = ModelCheckpoint(
            filepath=os.path.join(model_path, "best_loss.hdf5"),
            monitor='val_loss',
            verbose=1,
            save_best_only=True
        )
        
        return [checkpoint_balanced_acc, checkpoint_balanced_acc_weights, checkpoint_latest, checkpoint_loss]

    def _create_csvlogger_callback(self, subdir):
        if not os.path.exists(self.history_folder):
            os.makedirs(self.history_folder)

        return CSVLogger(
            filename=os.path.join(
                self.history_folder, 
                self.model_name, 
                subdir, 
                "training.csv"
            ), 
            append=True
        )

    def _create_tensorboard_logger(self, subdir):
        if not os.path.exists(self.history_folder):
            os.makedirs(self.history_folder)
        return TensorBoard(
            log_dir=os.path.join(
                self.history_folder, 
                self.model_name, 
                subdir
            ), 
            histogram_freq=0
        )

    @property
    def model(self):
        """CNN Model"""
        raise NotImplementedError(
            '`model` property method has not been implemented in {}.'
            .format(type(self).__name__)
        )

    @property
    def model_name(self):
        """Name of the CNN Model"""
        raise NotImplementedError(
            '`model_name` property method has not been implemented in {}.'
            .format(type(self).__name__)
        )
