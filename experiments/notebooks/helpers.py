import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils import get_hyperparameters_from_str 

def get_models_info(history_folder, model_name):
    d=os.path.join("..", history_folder, model_name)
    hyperparameter_combinations = [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    models_info = []

    for i, combination in enumerate(hyperparameter_combinations): 
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