import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from lesion_classifier import LesionClassifier
from base_model_param import BaseModelParam

class VanillaClassifier(LesionClassifier):
    """ 
        NOT IMPLEMENTED!!! Model trained from scratch for skin lesion classification
    """

    @property
    def model(self):
        return self._model

    @property
    def model_name(self):
        return self._model_name

    @staticmethod
    def preprocess_input(x, **kwargs):
        """Preprocesses a numpy array encoding a batch of images.
        # Arguments
            x: a 4D numpy array consists of RGB values within [0, 255].
        # Returns
            Preprocessed array.
        """
        return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)