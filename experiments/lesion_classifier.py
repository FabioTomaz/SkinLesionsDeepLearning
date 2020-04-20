import math
import os
import tempfile
import pandas as pd
import numpy as np
from data.augmentations import CustomPipeline, get_augmentation_group
from image_iterator import ImageIterator
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.regularizers import l2
from keras_numpy_backend import softmax

import random
import datetime

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


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
        rescale=True, 
        preprocessing_func=None, 
        class_weight=None,
        num_classes=None, 
        image_paths_train=None, 
        categories_train=None, 
        image_paths_val=None, 
        categories_val=None,
        online_data_augmentation_group=1
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
        self.preprocessing_func = preprocessing_func
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.image_paths_train = image_paths_train
        self.categories_train = categories_train
        self.image_paths_val = image_paths_val
        self.categories_val = categories_val
        self.online_data_augmentation_group = online_data_augmentation_group
        
        self.log_date = datetime.datetime.now().isoformat()

        self.aug_pipeline_train = LesionClassifier.create_aug_pipeline(
            self.online_data_augmentation_group,
            self.input_size,
            rescale
        )

        print('Image Augmentation Pipeline for Training Set')
        self.aug_pipeline_train.status()

        self.aug_pipeline_val = LesionClassifier.create_aug_pipeline(
            0,
            self.input_size,
            rescale
        )
        print('Image Augmentation Pipeline for Validation Set')
        self.aug_pipeline_val.status()

        self.generator_train, self.generator_val = self._create_image_generator()


    def add_regularization(self, model, regularizer=0.0001):
        regularizer = l2(regularizer)
        for layer in model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    print("L2 >> " + str(getattr(layer, attr)))
                    setattr(layer, attr, regularizer)

        # When we change the layers attributes, the change only happens in the model config file
        model_json = model.to_json()

        # Save the weights before reloading the model.
        tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
        model.save_weights(tmp_weights_path)

        # load the model from the config
        model = model_from_json(model_json)
        
        # Reload the model weights
        model.load_weights(tmp_weights_path, by_name=True)
        return model


    @staticmethod
    def create_aug_pipeline(data_aug_group, input_size, rescale):
        """Image Augmentation Pipeline for Training Set."""

        pipeline = CustomPipeline()

        data_aug_list = get_augmentation_group(
            data_aug_group, 
            input_size, 
            center=True, 
            resize=rescale
        )

        for aug in data_aug_list:
            pipeline.add_operation(aug)

        return pipeline


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
        workers=1
    ):
        
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
        intermediate_layer_model = Model(
            inputs=model.input,
            outputs=model.get_layer('dense_pred').output
        )
        logits = intermediate_layer_model.predict_generator(
            generator, 
            verbose=1, 
            workers=workers
        )

        softmax_probs = softmax(logits).astype(float) # explicitly convert softmax values to floating point because 0 and 1 are invalid, but 0.0 and 1.0 are valid

        # softmax probabilities
        df_softmax = pd.DataFrame(softmax_probs, columns=category_names)
        if y_col in df.columns:
            df_softmax[y_col] = df[y_col].to_numpy()
        df_softmax['pred_'+y_col] = np.argmax(softmax_probs, axis=1)
        df_softmax.insert(0, id_col, df[id_col].to_numpy())

        return df_softmax

    def _create_image_generator(self):
        ### Training Image Generator
        generator_train = ImageIterator(
            image_paths=self.image_paths_train,
            labels=self.categories_train,
            augmentation_pipeline=self.aug_pipeline_train,
            batch_size=self.batch_size,
            shuffle=True,
            rescale=None,
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
            rescale=None,
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
