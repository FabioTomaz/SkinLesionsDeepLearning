from typing import NamedTuple
from types import FunctionType
# from keras.applications.densenet import preprocess_input as preprocess_input_densenet
# from keras_applications.resnext import preprocess_input as preprocess_input_resnext
from keras.applications.xception import preprocess_input as preprocess_input_xception
from keras.applications.nasnet import preprocess_input as preprocess_input_nasnet
from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
from utils import preprocess_input as preprocess_input_trainset, preprocess_input_2 as preprocess_input_trainset_2

BaseModelParam = NamedTuple('BaseModelParam', [
    ('module_name', str),
    ('class_name', str),
    ('input_size', tuple),
    ('preprocessing_func', FunctionType)
])

def get_transfer_model_param_map():
    """MODELS"""
    models = {
        'DenseNet201': BaseModelParam(module_name='keras.applications.densenet',
                                      class_name='DenseNet201',
                                      input_size=(224, 224),
                                      preprocessing_func=preprocess_input_trainset),
        'Xception': BaseModelParam(module_name='keras.applications.xception',
                                   class_name='Xception',
                                   input_size=(299, 299),
                                   preprocessing_func=preprocess_input_xception),
        'NASNetLarge': BaseModelParam(module_name='keras.applications.nasnet',
                                      class_name='NASNetLarge',
                                      input_size=(331, 331),
                                      preprocessing_func=preprocess_input_nasnet),
        'InceptionResNetV2': BaseModelParam(module_name='keras.applications.inception_resnet_v2',
                                            class_name='InceptionResNetV2',
                                            input_size=(299, 299),
                                            preprocessing_func=preprocess_input_inception_resnet_v2),
        'VGG16': BaseModelParam(module_name='keras.applications.vgg16',
                                      class_name='VGG16',
                                      input_size=(224, 224),
                                      preprocessing_func=preprocess_input_trainset),                  
        'VGG19': BaseModelParam(module_name='keras.applications.vgg19',
                                      class_name='VGG19',
                                      input_size=(299, 299),
                                      preprocessing_func=preprocess_input_trainset),
        'EfficientNetB0': BaseModelParam(module_name='efficientnet.keras',
                                      class_name='EfficientNetB0',
                                      input_size=(224, 224),
                                      preprocessing_func=preprocess_input_trainset), 
        'EfficientNetB1': BaseModelParam(module_name='efficientnet.keras',
                                      class_name='EfficientNetB1',
                                      input_size=(240, 240),
                                      preprocessing_func=preprocess_input_trainset), 
        'EfficientNetB2': BaseModelParam(module_name='efficientnet.keras',
                                      class_name='EfficientNetB2',
                                      input_size=(260, 260),
                                      preprocessing_func=preprocess_input_trainset), 
        'EfficientNetB3': BaseModelParam(module_name='efficientnet.keras',
                                      class_name='EfficientNetB3',
                                      input_size=(300, 300),
                                      preprocessing_func=preprocess_input_trainset), 
        'EfficientNetB4': BaseModelParam(module_name='efficientnet.keras',
                                      class_name='EfficientNetB4',
                                      input_size=(380, 380),
                                      preprocessing_func=preprocess_input_trainset), 
        'EfficientNetB5': BaseModelParam(module_name='efficientnet.keras',
                                      class_name='EfficientNetB5',
                                      input_size=(456, 456),
                                      preprocessing_func=preprocess_input_trainset), 
        'EfficientNetB6': BaseModelParam(module_name='efficientnet.keras',
                                      class_name='EfficientNetB6',
                                      input_size=(456, 456),
                                      preprocessing_func=preprocess_input_trainset),                                                                                           
    }
    return models
