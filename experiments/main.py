import os
from utils import ensemble_predictions, ensemble_predictions_k_fold, formated_hyperparameter_str, get_gpu_index, save_prediction_results, apply_unknown_threshold, formated_hyperparameters
print("GPU DEVICE: " + str(get_gpu_index()))
os.environ["CUDA_VISIBLE_DEVICES"]=str(get_gpu_index())
import pandas as pd
import json
import argparse
import datetime
import efficientnet.tfkeras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras import utils as tfutils
from data.data_loader import load_isic_training_data, load_isic_training_and_out_dist_data, train_validation_split, compute_class_weight_dict, get_dataframe_from_img_folder
from transfer_learn_classifier import TransferLearnClassifier
from metrics import balanced_accuracy
from base_model_param import get_transfer_model_param_map
from lesion_classifier import LesionClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from typing import NamedTuple
import numpy as np
from odin import compute_out_of_distribution_score


UNKNOWN_THRESHOLDS=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

ModelParameters = NamedTuple('ModelParameters', [
    ('batch_size', int),
    ('fe_epochs', int),
    ('ft_epochs', int),
    ('felr', float),
    ('ftlr', float),
    ('lmbda', float),
    ('dropout', float),
    ('max_queue_size', int),
    ('offline_dg_group', int),
    ('online_dg_group', int),
    ('samples', int),
    ('balanced', int)
])

def train_transfer_learning(
    model_param, 
    df_train, 
    df_val, 
    num_classes, 
    class_weight_dict, 
    model_folder,
    parameters,
    k_split=0
):
    # Create classifier
    classifier = TransferLearnClassifier(
        model_folder=model_folder,
        base_model_param=model_param,
        fc_layers=[512], # e.g. [512]
        num_classes=num_classes,
        image_data_format=K.image_data_format(),
        metrics=[balanced_accuracy(num_classes), 'accuracy'],
        class_weight=class_weight_dict,
        image_paths_train=df_train['path'].tolist(),
        categories_train=tfutils.to_categorical(df_train['category'], num_classes=num_classes),
        image_paths_val=df_val['path'].tolist(),
        categories_val=tfutils.to_categorical(df_val['category'], num_classes=num_classes),
        parameters=parameters
    )
    print("Begin to train {}".format(model_param.class_name))
    classifier.train(k_split=k_split, workers=os.cpu_count())
    del classifier
    K.clear_session()


def train(
    base_model_params, 
    df_ground_truth,  
    k_folds,
    num_classes, 
    class_weight_dict, 
    model_folder,
    parameters
):    
    for model_param in base_model_params:
        if k_folds > 0:
            # K fold cross validation (stratified per category)
            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=1) 
            df_category = df_ground_truth.category
            k_split=0
            for train_index, val_index in kf.split(np.zeros(len(df_category)), df_category):
                df_train = df_ground_truth.loc[train_index]
                df_val = df_ground_truth.loc[val_index]
                train_transfer_learning(
                    model_param, 
                    df_train, 
                    df_val, 
                    num_classes, 
                    class_weight_dict,  
                    model_folder,
                    parameters,
                    k_split=k_split
                )
                k_split=k_split+1
        else:
            # Train/Validation split
            df_train, df_val = train_validation_split(df_ground_truth)
            train_transfer_learning(
                model_param, 
                df_train, 
                df_val, 
                num_classes, 
                class_weight_dict, 
                model_folder,
                parameters
            )


def handle_unknown(
    model,
    model_params,
    parameters,
    df_softmax,
    df_test,
    unknown_method,
    category_names=["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"],
    unknown_category="UNK",
    model_folder="models",
    pred_result_folder_test="test_predict_results",
    prediction_label="",
    postfix="best_balanced_acc"
):
    if unknown_method==0:
        # BASELINE (ignore unknown class)
        print("UNKNOWN CLASS IGNORED")
        save_prediction_results(
            df_softmax,
            model_params.class_name,
            pred_result_folder_test=pred_result_folder_test,
            parameters=parameters,
            prediction_label=os.path.join(
                prediction_label,
                "no_unknown"
            ),
            postfix=postfix
        )
    # Save model predictions 
    if unknown_method==1:
        # THRESHOLD
        print("HANDLING UNKNOWN CLASS WITH THRESHOLDING")
        # Compute softmax scores for unknown thresholds
        df_softmax_dict = apply_unknown_threshold(
            df_softmax,
            category_names,
            'image',
            'category',
            UNKNOWN_THRESHOLDS,
            unknown_category
        )
        # Save results
        for unknown_thresh, df_softmax in df_softmax_dict.items():
            save_prediction_results(
                df_softmax,
                model_params.class_name,
                prediction_label=os.path.join(
                    prediction_label,
                    f"unknown_{unknown_thresh}" if unknown_thresh != 1.0 else "no_unknown"
                ),
                pred_result_folder_test=pred_result_folder_test,
                parameters=parameters,
                postfix=postfix
            )
        df_softmax=df_softmax_dict[0.4]
    elif unknown_method==2:
        # ODIN
        print("HANDLING UNKNOWN CLASS WITH ODIN")
        # Compute Out-of-Distribution scores
        df_score = compute_out_of_distribution_score(
            model,
            model_params, 
            df_test, 
            len(category_names), 
            parameters
        )
        # Merge ensemble predictions with out-of-Distribution scores
        df_softmax.insert(
            loc=9, 
            column=unknown_category, 
            value=df_score['out_dist_score']
        )
        df_softmax['pred_category'] = np.argmax(np.array(df_softmax.iloc[:,1:10]), axis=1)
        save_prediction_results(
            df_softmax,
            model_params.class_name,
            prediction_label=os.path.join(
                prediction_label,
                f"odin"
            ),
            pred_result_folder_test=pred_result_folder_test,
            parameters=parameters,
            postfix=postfix
        )
    elif unknown_method==3:
        # OUTLIER CLASS
        print("HANDLING UNKNOWN CLASS WITH OUTLIER CLASS")
    else:
        print("UNKNOWN METHOD NOT IMPLEMENTED!")
    
    return df_softmax

def predidct_test(
    model_folder,
    test_image_folder,
    pred_result_folder_test,
    models_to_predict,
    category_names,
    unknown_method,
    unknown_category, 
    postfix,
    parameters,
    k_folds=0
):
    os.makedirs(pred_result_folder_test, exist_ok=True)
    df_test = get_dataframe_from_img_folder(test_image_folder, has_path_col=True)
    df_test.drop(columns=['path']).to_csv(os.path.join(pred_result_folder_test, 'ISIC_2019_Test.csv'), index=False)
    
    hyperparameter_str = formated_hyperparameters(parameters)

    model_kfold_folders=range(k_folds)
    if k_folds == 0:
        model_kfold_folders=[0]

    for model_to_predict in models_to_predict:
        for k_fold in model_kfold_folders:
            model_filepath = os.path.join(
                model_folder, 
                model_to_predict.class_name,
                hyperparameter_str,
                "{}.hdf5".format(postfix)
            )

            if not os.path.exists(model_filepath):
                model_filepath = os.path.join(
                    model_folder, 
                    model_to_predict.class_name,
                    hyperparameter_str,
                    str(k_fold),
                    "{}.hdf5".format(postfix)
                )

            if os.path.exists(model_filepath):
                print("===== Predict test data using \"{}_{}\" with \"{}\" model =====".format(model_to_predict.class_name, k_fold, postfix))
                
                # model = load_model(
                #     filepath=model_filepath, 
                #     custom_objects={'balanced_accuracy': balanced_accuracy(len(category_names))}
                # )

                # df_softmax = LesionClassifier.predict_dataframe(
                #     model=model, 
                #     df=df_test,
                #     category_names=category_names,
                #     augmentation_pipeline=LesionClassifier.create_aug_pipeline(0, model_to_predict.input_size, True),
                #     preprocessing_function=model_to_predict.preprocessing_func,
                #     batch_size=parameters.batch_size,
                #     workers=os.cpu_count(),
                # )

                df_softmax = pd.read_csv(
                    "/home/fmts/msc/experiments/test_predict_results/DenseNet201/balanced_1-samples_82400-feepochs_2-ftepochs_100-felr_0.001000-ftlr_0.000100-lambda_None-dropout_None-batch_16-dggroup_11/0/no_unknown/best_balanced_acc.csv"
                ) 
                df_softmax = handle_unknown(
                    None,
                    model_to_predict,
                    parameters,
                    df_softmax,
                    df_test,
                    unknown_method,
                    prediction_label=str(k_fold)
                )                

                #del model
                K.clear_session()
            else:
                print("\"{}\" doesn't exist".format(model_filepath))
                return

        if k_folds>0:
            print("===== Ensembling predictions from {} k folds of {}\" model =====".format(k_folds, model_to_predict.class_name))
            # Ensemble Models' k fold predictions
            df_softmax = ensemble_predictions_k_fold(
                result_folder=pred_result_folder_test, 
                parameters=parameters, 
                category_names=category_names, 
                model_name=model_to_predict.class_name, 
                postfix=postfix,
                k_folds=k_folds
            )            

        # Save ensemble predictions 
        save_prediction_results(
            df_softmax,
            model_to_predict.class_name,
            pred_result_folder_test=pred_result_folder_test,
            parameters=parameters,
            postfix=postfix
        )   

    if (len(models_to_predict)>1):
        model_names = [model.class_name for model in models_to_predict]

        print("===== Ensembling predictions using from {} models using {} =====".format(model_names, postfix))
                
        # Ensemble Models' Predictions on Test Data
        df_ensemble = ensemble_predictions(
            result_folder=pred_result_folder_test, 
            parameters=parameters, 
            category_names=category_names, 
            model_names=model_names, 
            postfix=postfix
        )

        # Save ensemble predictions 
        save_prediction_results(
            df_ensemble,
            "-".join([f"{model_name}_{k_folds}" for model_name in sorted(model_names)]),
            pred_result_folder_test=pred_result_folder_test
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ISIC-2019 Skin Lesion Classifier')
    parser.add_argument('data', metavar='DIR', help='path to data folder')
    parser.add_argument('--testfolder', metavar='DIR', help='path to test folder', default=None)
    parser.add_argument('--kfolds', type=int, help='Number of folds for k-fold cross validation (default: %(default)s)', default=0)
    parser.add_argument('--batchsize', type=int, help='Batch size (default: %(default)s)', default=32)
    parser.add_argument('--felr', type=float, help='Feature extractor learning rate (default: %(default)s)', default=1e-3)
    parser.add_argument('--ftlr', type=float, help='Fine tuning learning rate (default: %(default)s)', default=1e-5)
    parser.add_argument('--dropout', type=float, help='Dropout rate (default: %(default)s)', default=None)    
    parser.add_argument('--l2', type=float, help='l2 regularization parameter (default: %(default)s)', default=None)    
    parser.add_argument('--maxqueuesize', type=int, help='Maximum size for the generator queue (default: %(default)s)', default=10)
    parser.add_argument('--feepochs', type=int, help='Number of weight initialization epochs (default: %(default)s)', default=2)    
    parser.add_argument('--ftepochs', type=int, help='Number of fine tuning epochs (default: %(default)s)', default=100)
    parser.add_argument('--model', 
                        dest='models', 
                        nargs='*', 
                        choices=[
                            'DenseNet121',
                            'DenseNet169',
                            'DenseNet201', 
                            'InceptionV3', 
                            'InceptionResNetV2', 
                            'VGG16', 
                            'VGG19', 
                            "EfficientNetB0", 
                            "EfficientNetB1",
                            "EfficientNetB2",
                            "EfficientNetB3",
                            "EfficientNetB4",
                            "ResNet50",
                            "ResNet101",
                            "ResNet152",
                        ], 
                        help='Models',
                        default=["DenseNet201", "EfficientNetB2", "InceptionResNetV2"])
    parser.add_argument('--training', dest='training', action='store_true', help='Train models')
    parser.add_argument('--predtest', dest='predtest', action='store_true', help='Predict the test data')
    parser.add_argument('--online-data-augmentation-group', dest='online_dg_group', default=1, type=int)
    parser.add_argument('--unknown', type=int, default=0, help='Method to deal with unknown class')
    parser.add_argument(
        '--predtestresultfolder', 
        help='Name of the prediction result folder for test data (default: %(default)s)', 
        default='test_predict_results'
    )
    parser.add_argument('--modelfolder', help='Name of the model folder (default: %(default)s)', default='models')
    parser.add_argument('--postfix', help='Postfix name (default: %(default)s)', default='best_balanced_acc', choices=['best_balanced_acc', 'best_loss', 'latest'])

    args = parser.parse_args()

    # Write command to a file
    with open('Cmd_History.txt', 'a') as f:
        f.write("{}\t{}\n".format(str(datetime.datetime.utcnow()), str(args)))

    data_folder = args.data

    # ISIC data
    training_image_folder = os.path.join(data_folder, 'ISIC_2019_Training_Input')
    test_image_folder = args.testfolder if args.testfolder is not None else os.path.join(data_folder, 'ISIC_2019_Test_Input')

    # Ground truth
    ground_truth_file = os.path.join(data_folder, 'ISIC_2019_Training_GroundTruth.csv')
    df_ground_truth, known_category_names, unknown_category_name = load_isic_training_data(training_image_folder, ground_truth_file)
    class_weight_dict = compute_class_weight_dict(df_ground_truth)

    offline_dg_group = 1
    # Read Data METADATA File
    with open(os.path.join(data_folder, 'metadata.json')) as f:
        data_metadata = json.load(f)
        offline_dg_group = int(data_metadata["data_augmentation_group"])

    # Set parameters object
    parameters = ModelParameters(
        batch_size=args.batchsize,
        dropout=args.dropout,
        fe_epochs=args.feepochs,
        ft_epochs=args.ftepochs,
        felr = args.felr,
        ftlr = args.ftlr,        
        lmbda=args.l2,
        max_queue_size = args.maxqueuesize,
        offline_dg_group=offline_dg_group,
        online_dg_group=args.online_dg_group,
        samples=len(df_ground_truth['path'].tolist()),
        balanced=all(round(value, 2) == 1 for value in class_weight_dict.values())
    )

    # Train models by Transfer Learning
    model_param_map = get_transfer_model_param_map() 

    # Models for training/testing
    base_model_params = [model_param_map[x] for x in args.models]

    # Train Models
    if args.training:
        train(
            base_model_params, 
            df_ground_truth,  
            args.kfolds,
            len(known_category_names), 
            class_weight_dict, 
            args.modelfolder,
            parameters
        )

    # Predict Test Data
    if args.predtest:
        predidct_test(
            args.modelfolder,
            test_image_folder,
            args.predtestresultfolder,
            base_model_params,
            known_category_names,
            args.unknown, 
            unknown_category_name,
            args.postfix,
            parameters,
            k_folds=args.kfolds
        )
