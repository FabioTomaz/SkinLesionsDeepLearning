import os,sys,inspect
import pandas as pd
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils import get_hyperparameters_from_str 
from data.data_loader import load_isic_training_data
from collections import Counter


def read_models_info(history_folder_name, pred_test_folder_name):
    models_info = []

    history_folder=os.path.join("..", history_folder_name)
    pred_test_folder=os.path.join("..", pred_test_folder_name)

    model_names = [o for o in os.listdir(history_folder) if os.path.isdir(os.path.join(history_folder,o))]
    for model_name in model_names:
        log_dir = os.path.join(history_folder, model_name)
        hyperparameter_combinations = [o for o in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir,o))]

        for i, combination in enumerate(hyperparameter_combinations): 
            hyperparameters = get_hyperparameters_from_str(combination)
            file_path = os.path.join(history_folder, model_name, combination, "0", "training.csv")
            pred_test_path_0 = os.path.join(pred_test_folder, model_name, combination, "0")
            pred_test_path = os.path.join(pred_test_folder, model_name, combination)

            if len(hyperparameters)>1 and os.path.exists(file_path):
                model_info = {
                    "model": model_name,
                    "hyperparameters": hyperparameters, 
                    "log": file_path,
                    "pred_test_0": pred_test_path_0 if os.path.exists(pred_test_path_0) else None,
                    "pred_test": pred_test_path if os.path.exists(pred_test_path) else None
                }
                models_info.append(model_info)
    
    return models_info


def filter_models_info(models_info, models=None, parameters=None):
    models_info_list = []
    for model_info in models_info:
        add=True
        if models is not None and model_info["model"] not in models:
            add=False  
        if parameters is not None:
            for key, parameter_filter in parameters.items():
                if key in model_info["hyperparameters"]:
                    model_parameter = model_info["hyperparameters"][key]
                    if isinstance(parameter_filter, list):
                        if float(model_parameter) not in [None if i == "None" else float(i) for i in parameter_filter]:
                            add=False
                            break
                    elif parameter_filter is None or model_parameter == "None":
                        if str(parameter_filter) != model_parameter:
                            add=False
                            break
                    else:
                        try:
                            parameter_filter_float=float(parameter_filter)
                            if float(model_parameter) != parameter_filter_float:
                                add=False
                                break
                        except ValueError:
                            continue
                else:
                    add=False

        if add is True:
            models_info_list.append(model_info)
    return models_info_list


def read_ensembles(
    pred_test_folder,
    contains_models=None
):
    models_info = []

    ensemble_names = [o for o in os.listdir(pred_test_folder) if os.path.isdir(os.path.join(pred_test_folder,o)) and "-" in o]
    for ensemble_name in ensemble_names:
        pred_test_path=os.path.join(pred_test_folder, ensemble_name)
        models=ensemble_name.split("-")
        if contains_models is not None:
            add=True
            for model_to_contain in contains_models:
                if model_to_contain not in models:
                    add=False
            if add is False:
                continue 

        model_info = {
            "models": ensemble_name.split("-"),
            "pred_test": pred_test_path,
        }
        models_info.append(model_info)
    
    return models_info


def get_count_per_category(data_folder, test=False):
    image_folder = os.path.join(data_folder, 'ISIC_2019_Test_Input' if test else 'ISIC_2019_Training_Input')
    ground_truth_file = os.path.join(data_folder, 'ISIC_2019_Test_GroundTruth.csv' if test else 'ISIC_2019_Training_GroundTruth.csv')

    df_test_ground_truth, known_category_names, unknown_category_name = load_isic_training_data(
        image_folder, 
        ground_truth_file,
        test=True
    )

    all_category_names = known_category_names + [unknown_category_name]
    count_per_category = Counter(df_test_ground_truth['category'])

    return all_category_names, count_per_category, df_test_ground_truth


def get_log_metric(log_file, metric="val_balanced_accuracy", criteria="val_balanced_accuracy"):
    df = pd.read_csv(log_file)
    return df.iloc[df[criteria].idxmax()][metric]


def get_test_metric(df_pred, df_truth, metric_func):
    df = pd.merge(df_pred, df_truth, on='image')
    y_true = df['category']
    y_pred = df['pred_category']
    return metric_func(y_true, y_pred)