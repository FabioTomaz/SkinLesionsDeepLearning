from importlib import import_module
from lesion_classifier import LesionClassifier
from base_model_param import BaseModelParam
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Activation, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
#from tensorflow import distribute
from utils import formated_hyperparameters
import os


class TransferLearnClassifier(LesionClassifier):
    """Skin lesion classifier based on transfer learning.

    # Arguments
        base_model_param: Instance of `BaseModelParam`.
    """

    def __init__(
        self, 
        model_folder, 
        history_folder,
        base_model_param, 
        fc_layers=None, 
        num_classes=None, 
        image_data_format=None, 
        metrics=None,
        class_weight=None, 
        image_paths_train=None, 
        categories_train=None, 
        image_paths_val=None, 
        categories_val=None,
        parameters=None,
        rescale=True
    ):
        self._model_name = base_model_param.class_name
        self.metrics = metrics

        if parameters is None:
            raise ValueError('parameters cannot be None!!')
        self.parameters = parameters

        if image_data_format is None:
            image_data_format = K.image_data_format()
            
        if image_data_format == 'channels_first':
            input_shape = (3, base_model_param.input_size[0], base_model_param.input_size[1])
        else:
            input_shape = (base_model_param.input_size[0], base_model_param.input_size[1], 3)

        # Dynamically get the class name of base model
        module = import_module(base_model_param.module_name)
        class_ = getattr(module, base_model_param.class_name)

        #with self.mirrored_strategy.scope():
        # create an instance of base model which is pre-trained on the ImageNet dataset.
        self._base_model = class_(include_top=False, weights='imagenet', input_shape=input_shape)

        # Freeze all layers in the base model
        for layer in self._base_model.layers:
            layer.trainable = False

        x = self._base_model.output
        x = GlobalAveragePooling2D()(x)
        # Add fully connected layers
        if fc_layers is not None:
            for fc in fc_layers:
                x = Dense(fc, activation='relu')(x)
                if self.parameters.dropout is not None:
                    x = Dropout(rate=self.parameters.dropout)(x)

        # Final dense layer and softmax activation layer
        x = Dense(num_classes, name='dense_pred')(x)
        predictions = Activation('softmax', name='probs')(x)

        # Create the model
        self._model = Model(inputs=self._base_model.input, outputs=predictions)

        if self.parameters.lmbda is not None:
            self._model = self.add_regularization(self._model, self.parameters.lmbda)

        # Compile the model
        self._model.compile(
            optimizer=Adam(lr=self.parameters.felr), 
            loss='categorical_crossentropy', 
            metrics=self.metrics
        )

        super().__init__(
            model_folder, 
            history_folder,
            base_model_param.input_size, 
            parameters,
            preprocessing_func=base_model_param.preprocessing_func, 
            class_weight=class_weight, 
            num_classes=num_classes,
            image_data_format=image_data_format, 
            image_paths_train=image_paths_train, 
            categories_train=categories_train,
            image_paths_val=image_paths_val, 
            categories_val=categories_val,
            rescale=rescale
        )


    def train(self, k_split=0, workers=1):
        # Sub pre-trained model folder
        model_subdir = os.path.join(formated_hyperparameters(self.parameters), str(k_split))

        # Checkpoint Callbacks
        checkpoints = super()._create_checkpoint_callbacks(model_subdir)

        # Reduce learning rate when the validation loss has stopped improving.
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=1e-7, verbose=1)

        # Stop training when the validation loss has stopped improving.
        early_stop = EarlyStopping(monitor='val_loss', patience=16, verbose=1)

        # Callback that streams epoch results to a csv file.
        csv_logger = super()._create_csvlogger_callback(model_subdir)

        # Callback that streams epoch results to tensorboard
        tensorboard_logger = super()._create_tensorboard_logger(model_subdir)

        if(self.parameters.fe_epochs>0):
            ### Feature extraction
            self._model.fit(
                self.generator_train,
                class_weight=self.class_weight,
                max_queue_size=self.parameters.max_queue_size,
                workers=workers,
                use_multiprocessing=True,
                steps_per_epoch=len(self.image_paths_train)//self.parameters.batch_size,
                epochs=self.parameters.fe_epochs,
                verbose=1,
                callbacks=(checkpoints + [reduce_lr, early_stop, csv_logger, tensorboard_logger]),
                validation_data=self.generator_val,
                validation_steps=len(self.image_paths_val)//self.parameters.batch_size,
                shuffle=False
            )
        else:
            print('===== No weight initialization =====')

        if(self.parameters.ft_epochs>0):
            ### Fine tuning. It should only be attempted after you have trained the top-level classifier with the pre-trained model set to non-trainable.
            print('===== Unfreeze the base model =====')
            for layer in self._base_model.layers:
                layer.trainable = True

            #with self.mirrored_strategy.scope():
            # Compile the model
            self._model.compile(
                optimizer=Adam(lr=self.parameters.ftlr), 
                loss='categorical_crossentropy', 
                metrics=self.metrics
            )
            self._model.summary()

            # Re-create Checkpoint Callbacks
            checkpoints = super()._create_checkpoint_callbacks(model_subdir)

            self.generator_train.reset()
            self.generator_val.reset()
            
            self._model.fit(
                self.generator_train,
                class_weight=self.class_weight,
                max_queue_size=self.parameters.max_queue_size,
                workers=workers,
                use_multiprocessing=True,
                steps_per_epoch=len(self.image_paths_train)//self.parameters.batch_size,
                epochs=self.parameters.ft_epochs,
                verbose=1,
                callbacks=(checkpoints + [reduce_lr, early_stop, csv_logger, tensorboard_logger]),
                validation_data=self.generator_val,
                validation_steps=len(self.image_paths_val)//self.parameters.batch_size,
                initial_epoch=self.parameters.fe_epochs,
                shuffle=False
            )
        else:
            print('===== No fine tuning =====')

    @property
    def model(self):
        return self._model

    @property
    def model_name(self):
        return self._model_name
