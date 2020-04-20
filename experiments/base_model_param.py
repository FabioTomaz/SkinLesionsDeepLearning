from typing import NamedTuple
from types import FunctionType
# from keras.applications.densenet import preprocess_input as preprocess_input_densenet

from utils import preprocess_input as preprocess_input_trainset

BaseModelParam = NamedTuple('BaseModelParam', [
    ('module_name', str),
    ('class_name', str),
    ('input_size', tuple),
    ('preprocessing_func', FunctionType),
    ('feepochs', int),
    ('felr', float),
    ('ftlr', float),
    ('dropout', float),
    ('batch', int),
])

def get_transfer_model_param_map():
    """MODELS"""
    models = {
        'DenseNet121': BaseModelParam(
            module_name='tensorflow.keras.applications.densenet',
            class_name='DenseNet121',
            input_size=(224, 224),
            preprocessing_func=preprocess_input_trainset,
            feepochs=2,
            felr=1e-3,
            ftlr=1e-3,
            dropout=None,
            batch=16
        ),
        'DenseNet169': BaseModelParam(
            module_name='tensorflow.keras.applications.densenet',
            class_name='DenseNet169',
            input_size=(224, 224),
            preprocessing_func=preprocess_input_trainset,
            feepochs=2,
            felr=1e-3,
            ftlr=1e-3,
            dropout=None,
            batch=16
        ),
        'DenseNet201': BaseModelParam(
            module_name='tensorflow.keras.applications.densenet',
            class_name='DenseNet201',
            input_size=(224, 224),
            preprocessing_func=preprocess_input_trainset,
            feepochs=2,
            felr=1e-3,
            ftlr=1e-3,
            dropout=None,
            batch=16
        ),
        'InceptionV3': BaseModelParam(
            module_name='tensorflow.keras.applications.inception_v3',
            class_name='InceptionV3',
            input_size=(299, 299),
            preprocessing_func=preprocess_input_trainset,
            feepochs=2,
            felr=1e-3,
            ftlr=1e-3,
            dropout=None,
            batch=16
        ),
        'InceptionResNetV2': BaseModelParam(
            module_name='tensorflow.keras.applications.inception_resnet_v2',
            class_name='InceptionResNetV2',
            input_size=(299, 299),
            preprocessing_func=preprocess_input_trainset,
            feepochs=2,
            felr=1e-3,
            ftlr=1e-3,
            dropout=None,
            batch=16
        ),
        'Xception': BaseModelParam(
            module_name='tensorflow.keras.applications.xception',
            class_name='Xception',
            input_size=(299, 299),
            preprocessing_func=preprocess_input_trainset,
            feepochs=2,
            felr=1e-3,
            ftlr=1e-3,
            dropout=None,
            batch=16
        ),
        'VGG16': BaseModelParam(
            module_name='tensorflow.keras.applications.vgg16',
            class_name='VGG16',
            input_size=(224, 224),
            preprocessing_func=preprocess_input_trainset,
            feepochs=2,
            felr=1e-3,
            ftlr=1e-3,
            dropout=None,
            batch=16
        ),                
        'VGG19': BaseModelParam(
            module_name='tensorflow.keras.applications.vgg19',
            class_name='VGG19',
            input_size=(299, 299),
            preprocessing_func=preprocess_input_trainset,
            feepochs=2,
            felr=1e-3,
            ftlr=1e-3,
            dropout=None,
            batch=16
        ),
        'EfficientNetB0': BaseModelParam(
            module_name='efficientnet.tfkeras',
            class_name='EfficientNetB0',
            input_size=(224, 224),
            preprocessing_func=preprocess_input_trainset,
            feepochs=2,
            felr=1e-3,
            ftlr=1e-3,
            dropout=None,
            batch=16
        ),
        'EfficientNetB1': BaseModelParam(
            module_name='efficientnet.tfkeras',
            class_name='EfficientNetB1',
            input_size=(240, 240),
            preprocessing_func=preprocess_input_trainset,
            feepochs=2,
            felr=1e-3,
            ftlr=1e-3,
            dropout=None,
            batch=16
        ),
        'EfficientNetB2': BaseModelParam(
            module_name='efficientnet.tfkeras',
            class_name='EfficientNetB2',
            input_size=(260, 260),
            preprocessing_func=preprocess_input_trainset,
            feepochs=2,
            felr=1e-3,
            ftlr=1e-3,
            dropout=None,
            batch=16
        ),
        'EfficientNetB3': BaseModelParam(
            module_name='efficientnet.tfkeras',
            class_name='EfficientNetB3',
            input_size=(300, 300),
            preprocessing_func=preprocess_input_trainset,
            feepochs=2,
            felr=1e-3,
            ftlr=1e-3,
            dropout=None,
            batch=16
        ), 
        'EfficientNetB4': BaseModelParam(
            module_name='efficientnet.tfkeras',
            class_name='EfficientNetB4',
            input_size=(380, 380),
            preprocessing_func=preprocess_input_trainset,
            feepochs=2,
            felr=1e-3,
            ftlr=1e-3,
            dropout=None,
            batch=16
        ),
        'ResNet50': BaseModelParam(
            module_name='tensorflow.keras.applications.resnet',
            class_name='ResNet50',
            input_size=(224, 224),
            preprocessing_func=preprocess_input_trainset,
            feepochs=2,
            felr=1e-3,
            ftlr=1e-3,
            dropout=None,
            batch=16
        ),
        'ResNet101': BaseModelParam(
            module_name='tensorflow.keras.applications.resnet',
            class_name='ResNet101',
            input_size=(224, 224),
            preprocessing_func=preprocess_input_trainset,
            feepochs=2,
            felr=1e-3,
            ftlr=1e-3,
            dropout=None,
            batch=16
        ),
        'ResNet152': BaseModelParam(
            module_name='tensorflow.keras.applications.resnet',
            class_name='ResNet152',
            input_size=(224, 224),
            preprocessing_func=preprocess_input_trainset,
            feepochs=2,
            felr=1e-3,
            ftlr=1e-3,
            dropout=None,
            batch=16
        ),                                                                                         
    }
    return models
