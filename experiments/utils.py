from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import os
import subprocess as sp
import cv2
from keras_numpy_backend import softmax


def logistic(x, x0=0, L=1, k=1):
    """ Calculate the value of a logistic function.

    # Arguments
        x0: The x-value of the sigmoid's midpoint.
        L: The curve's maximum value.
        k: The logistic growth rate or steepness of the curve.
    # References https://en.wikipedia.org/wiki/Logistic_function
    """

    return L / (1 + np.exp(-k*(x-x0)))


def formated_hyperparameter_str(
    feepochs,
    ftepochs,
    felr,
    ftlr,
    lmda,
    dropout,
    batch_size,
    samples,
    balanced,
    offline_data_augmentation_group=1,
    online_data_augmentation_group=1,
    patience=8
):
    felr_str = format(felr, 'f')
    ftlr_str = format(ftlr, 'f')
    dropout_str = "None" if dropout == None else format(dropout, 'f')
    l2_str = "None" if lmda == None else format(lmda, 'f')
    balanced_int = 1 if balanced is True else 0
    data_augmentation_group = str(offline_data_augmentation_group) + str(online_data_augmentation_group)
    return f'balanced_{balanced_int}-samples_{samples}-feepochs_{feepochs}-ftepochs_{ftepochs}-felr_{felr_str}-ftlr_{ftlr_str}-lambda_{l2_str}-dropout_{dropout_str}-batch_{batch_size}-dggroup_{data_augmentation_group}-patience_{patience}'


def formated_hyperparameters(parameters):
    return formated_hyperparameter_str(
        parameters.fe_epochs,
        parameters.ft_epochs,
        parameters.felr,
        parameters.ftlr,
        parameters.lmbda,
        parameters.dropout,
        parameters.batch_size,
        parameters.samples,
        parameters.balanced,
        parameters.offline_dg_group,
        parameters.online_dg_group,
        parameters.patience
    )


def get_hyperparameters_from_str(hyperparameter_str):
    hyperparameter_combination = hyperparameter_str.split("-")
    hyperparameters = {}
    for hyperparameter in hyperparameter_combination:
        split=hyperparameter.split("_")
        if(len(split)==2):
            #split_val = "0.0" if split[1]=="None" else split[1]
            hyperparameters[split[0]] = split[1]
    return hyperparameters


def get_gpu_index():
    """ Returns available gpu index or None if none is available
    """
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.total,memory.used --format=csv"
    gpu_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]

    memory_used_percent = []
    for i in gpu_info:
        memory_total = float(i.split(",")[0].split()[0])
        memory_used = float(i.split(",")[1].split()[0])
        memory_used_percent.append(float(memory_used)/float(memory_total)*100)

    memory_min_val = min(memory_used_percent)
    return memory_used_percent.index(memory_min_val) if memory_min_val < 10 else None


def path_to_tensor(img_path, size=(224, 224)):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=size)
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths, size=(224, 224)):
    list_of_tensors = [path_to_tensor(img_path, size) for img_path in img_paths]
    return np.vstack(list_of_tensors)


def calculate_mean_std(img_paths):
    """
    Calculate the image per channel mean and standard deviation.

    # References
        https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6
    """
    
    # Number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
    channel_num = 3
    pixel_num = 0 # store all pixel number in the dataset
    channel_sum = np.zeros(channel_num)
    channel_sum_squared = np.zeros(channel_num)

    for path in img_paths:
        im = cv2.imread(path) # image in M*N*CHANNEL_NUM shape, channel in BGR order
        im = im/255.
        pixel_num += (im.size/channel_num)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))
    
    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]
    
    return rgb_mean, rgb_std


def preprocess_input(x, data_format=None):
    """Preprocesses a numpy array encoding a batch of images. Each image is normalized by subtracting the mean and dividing by the standard deviation channel-wise.
    This function only implements the 'torch' mode which scale pixels between 0 and 1 and then will normalize each channel with respect to the training dataset of approach 1 (not include validation set).

    # Arguments
        x: a 3D or 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.
    # Returns
        Preprocessed array.
    # References
        https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
    """
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(K.floatx(), copy=False)

    # Mean and STD from ImageNet
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Mean and STD calculated over the Training Set
    # Mean:[0.6236094091893962, 0.5198354883713194, 0.5038435406338101]
    # STD:[0.2421814437693499, 0.22354427793687906, 0.2314805420919389]
    x /= 255.
    mean = [0.6236, 0.5198, 0.5038]
    std = [0.2422, 0.2235, 0.2315]
    
    np.mean(x, axis=(0, 1))
    np.std(x, axis=(0, 1))

    if data_format is None:
        data_format = K.image_data_format()

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x


def apply_unknown_threshold(
    df,
    category_names,
    id_col,
    y_col,
    unknown_thresholds,
    unknown_category
):
    df_softmax_dict = {}

    # convert softmax values to floating point to become valid
    original_softmax_probs = df.iloc[:,1:9].values 
    
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


def save_prediction_results(
    df_softmax,
    model_name,
    pred_result_folder_test="test_predict_results",
    parameters=None,
    prediction_label="no_unknown",
    postfix="best_balanced_acc"
):
    hyperparameter_str=""
    if parameters is not None:
        hyperparameter_str = formated_hyperparameters(parameters)
    # Save results (multiple thresholds and no threhold)
    pred_folder = os.path.join(
        pred_result_folder_test, 
        model_name,
        hyperparameter_str,
        prediction_label,
    )
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)

    df_softmax.to_csv(path_or_buf=os.path.join(pred_folder, f"{postfix}.csv"), index=False)
    df_softmax_no_pred_col = df_softmax.copy()
    del df_softmax_no_pred_col['pred_category']
    df_softmax_no_pred_col.to_csv(path_or_buf=os.path.join(pred_folder, f"{postfix}_no_pred_category.csv"), index=False)


def ensemble_predictions_k_fold(
    result_folder, 
    parameters,
    category_names, 
    model_name='DenseNet201',
    postfix='best_balanced_acc',
    k_folds=5
):
    hyperparameter_str = formated_hyperparameters(parameters)
    """ Ensemble predictions of different models. """
    # Load models' predictions
    df_dict = {
        fold : pd.read_csv(
            os.path.join(
                result_folder, 
                model_name, 
                hyperparameter_str, 
                str(fold),
                "no_unknown",
                f"{postfix}.csv"
            )
        ) for fold in range(k_folds)
    }

    # Copy the first model's predictions
    df_ensemble = df_dict[0].drop(columns=['pred_category'])

    # Add up predictions
    for category_name in category_names:
        for i in range(k_folds):
            df_ensemble[category_name] = df_ensemble[category_name] + df_dict[i][category_name]

    # Take average of predictions
    for category_name in category_names:
        df_ensemble[category_name] = df_ensemble[category_name] / k_folds

    # Ensemble Predictions
    df_ensemble['pred_category'] = np.argmax(np.array(df_ensemble.iloc[:,1:(1+len(category_names))]), axis=1)

    return df_ensemble


def ensemble_predictions(
    result_folder, 
    parameters,
    category_names, 
    model_names=['DenseNet201', 'ResNet152', 'EfficientNetB2'],
    postfix='best_balanced_acc'
):
    """ Ensemble predictions of different models. """

    hyperparameter_str = formated_hyperparameters(parameters)

    # Load models' predictions
    df_dict = {
        model_name : pd.read_csv(
            os.path.join(
                result_folder, 
                model_name, 
                hyperparameter_str, 
                "no_unknown",
                f"{postfix}.csv"
            )
        ) for i, model_name in enumerate(model_names)
    }

    # Check row number
    for i in range(1, len(model_names)):
        if len(df_dict[model_names[0]]) != len(df_dict[model_names[i]]):
            raise ValueError(
                f"Row numbers are inconsistent between {model_names[0]} and {model_names[i]}"
            )

    # Check whether values of image column are consistent
    for i in range(1, len(model_names)):
        inconsistent_idx = np.where(df_dict[model_names[0]].image != df_dict[model_names[i]].image)[0]
        if len(inconsistent_idx) > 0:
            raise ValueError(
                "{} values of image column are inconsistent between {} and {}".format(
                    len(inconsistent_idx), 
                    model_names[0], 
                    model_names[i]
                )
            )

    # Copy the first model's predictions
    df_ensemble = df_dict[model_names[0]].drop(columns=['pred_category'])

    # Add up predictions
    for category_name in category_names:
        for i in range(1, len(model_names)):
            df_ensemble[category_name] = df_ensemble[category_name] + df_dict[model_names[i]][category_name]

    # Take average of predictions
    for category_name in category_names:
        df_ensemble[category_name] = df_ensemble[category_name] / len(model_names)

    # Ensemble Predictions
    df_ensemble['pred_category'] = np.argmax(np.array(df_ensemble.iloc[:,1:(1+len(category_names))]), axis=1)

    return df_ensemble

