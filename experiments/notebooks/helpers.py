import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils import get_hyperparameters_from_str 
from data.data import load_isic_training_data
from collections import Counter


def get_models_info(history_folder, model_name):
    d=os.path.join("..", history_folder, model_name)
    hyperparameter_combinations = [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    models_info = []

    for _, combination in enumerate(hyperparameter_combinations): 
        hyperparameters = get_hyperparameters_from_str(combination)
        file_path = os.path.join("..", history_folder, model_name, combination, "training.csv")
        if(len(hyperparameters)>1 and os.path.exists(file_path)):
            model_info = {
                "log": file_path,
                "hyperparameters": hyperparameters, 
                "hyperparameters_dir": combination
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

    return all_category_names, count_per_category