import os
from utils import ensemble_predictions, formated_hyperparameter_str, get_gpu_index, save_prediction_results
print("GPU DEVICE: " + str(get_gpu_index()))
os.environ["CUDA_VISIBLE_DEVICES"]=str(get_gpu_index())

import json
import argparse
import datetime
# Import layers to load model
import efficientnet.tfkeras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras import utils
from data.data_loader import load_isic_training_data, load_isic_training_and_out_dist_data, train_validation_split, compute_class_weight_dict, get_dataframe_from_img_folder
from transfer_learn_classifier import TransferLearnClassifier
from metrics import balanced_accuracy
from base_model_param import get_transfer_model_param_map
from lesion_classifier import LesionClassifier
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from utils import apply_unknown_threshold

UNKNOWN_THRESHOLDS=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def train_transfer_learning(
    model_param, 
    df_train, 
    df_val, 
    num_classes, 
    class_weight_dict, 
    batch_size, 
    max_queue_size, 
    ft_epochs, 
    model_folder,
    fe_epochs,
    dropout,
    felr,
    ftlr,
    lmbda,
    offline_dg_group,
    online_dg_group,
    k_split=0
):
    workers = os.cpu_count()

    classifier = TransferLearnClassifier(
        model_folder=model_folder,
        base_model_param=model_param,
        fc_layers=[512], # e.g. [512]
        num_classes=num_classes,
        dropout=dropout, 
        feature_extract_epochs= fe_epochs,
        feature_extract_start_lr=felr, 
        fine_tuning_epochs= ft_epochs,
        l2=lmbda,
        fine_tuning_start_lr=ftlr,
        batch_size=batch_size,
        max_queue_size=max_queue_size,
        image_data_format=K.image_data_format(),
        metrics=[balanced_accuracy(num_classes), 'accuracy'],
        class_weight=class_weight_dict,
        image_paths_train=df_train['path'].tolist(),
        categories_train=utils.to_categorical(df_train['category'], num_classes=num_classes),
        image_paths_val=df_val['path'].tolist(),
        categories_val=utils.to_categorical(df_val['category'], num_classes=num_classes),
        offline_data_augmentation_group=offline_dg_group,
        online_data_augmentation_group=online_dg_group
    )
    classifier.model.summary()
    print("Begin to train {}".format(model_param.class_name))
    classifier.train(k_split=k_split, workers=workers)
    del classifier
    K.clear_session()


def train(
    base_model_params, 
    df_ground_truth,  
    k_folds,
    num_classes, 
    class_weight_dict, 
    batch_size, 
    max_queue_size, 
    ft_epochs, 
    model_folder,
    fe_epochs,
    dropout,
    felr,
    ftlr,
    lmbda,
    offline_dg_group,
    online_dg_group
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
                    batch_size, 
                    max_queue_size, 
                    ft_epochs, 
                    model_folder,
                    fe_epochs,
                    dropout,
                    felr,
                    ftlr,
                    lmbda,
                    offline_dg_group,
                    online_dg_group,
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
                batch_size, 
                max_queue_size, 
                ft_epochs, 
                model_folder,
                fe_epochs,
                dropout,
                felr,
                ftlr,
                lmbda,
                offline_dg_group,
                online_dg_group
            )


def predidct_test(
    model_folder,
    test_image_folder,
    pred_result_folder_test,
    models_to_predict,
    category_names,
    unknown_method,
    unknown_category, 
    samples, 
    balanced, 
    batch_size, 
    ft_epochs, 
    fe_epochs,
    dropout,
    felr,
    ftlr,
    lmbda,
    postfix,
    offline_dg_group,
    online_dg_group
):
    os.makedirs(pred_result_folder_test, exist_ok=True)
    df_test = get_dataframe_from_img_folder(test_image_folder, has_path_col=True)
    df_test.drop(columns=['path']).to_csv(os.path.join(pred_result_folder_test, 'ISIC_2019_Test.csv'), index=False)
    
    hyperparameter_str = formated_hyperparameter_str(
        fe_epochs,
        ft_epochs,
        felr,
        ftlr,
        lmbda,
        dropout,
        batch_size,
        samples,
        balanced,
        offline_dg_group,
        online_dg_group
    )
    
    for m in models_to_predict:
        model_filepath = os.path.join(
            model_folder, 
            m["model_name"],
            hyperparameter_str,
            "{}.hdf5".format(postfix)
        )

        if not os.path.exists(model_filepath):
            model_filepath = os.path.join(
                model_folder, 
                m["model_name"],
                hyperparameter_str,
                "0",
                "{}.hdf5".format(postfix)
            )

        if os.path.exists(model_filepath):
            print("===== Predict test data using \"{}_{}\" model =====".format(m['model_name'], postfix))
            
            model = load_model(
                filepath=model_filepath, 
                custom_objects={'balanced_accuracy': balanced_accuracy(len(category_names))}
            )

            df_softmax = LesionClassifier.predict_dataframe(
                model=model, 
                df=df_test,
                category_names=category_names,
#                unknown_category=unknown_category_name if args.unknown else None,
                augmentation_pipeline=LesionClassifier.create_aug_pipeline(0, m['input_size'], True),
                preprocessing_function=m['preprocessing_function'],
                batch_size=batch_size,
                workers=os.cpu_count(),
            )

            if unknown_method==1:
                df_softmax_dict = apply_unknown_threshold(
                    df_softmax,
                    category_names,
                    'image',
                    'category',
                    args.unknown,
                    unknown_category
                )
                    
                for unknown_thresh, df_softmax in df_softmax_dict.items():
                    # Save model predictions 
                    save_prediction_results(
                        df_softmax,
                        f"unknown_{unknown_thresh}" if unknown_thresh != 1.0 else "no_unknown",
                        pred_result_folder_test,
                        m["model_name"],
                        postfix,
                        hyperparameter_str=hyperparameter_str
                    )
            else:
                save_prediction_results(
                    df_softmax,
                    "no_unknown",
                    pred_result_folder_test,
                    m["model_name"],
                    postfix,
                    hyperparameter_str=hyperparameter_str
                )                

            del model
            K.clear_session()
        else:
            print("\"{}\" doesn't exist".format(model_filepath))

    if (len(models_to_predict)>1):
        print("===== Ensembling predictions using \"{}_{}\" model =====".format(models_to_predict, postfix))
        
        model_names = [model["model_name"] for model in models_to_predict]
        
        # Ensemble Models' Predictions on Test Data
        df_ensemble = ensemble_predictions(
            result_folder=pred_result_folder_test, 
            hyperparameter_str=hyperparameter_str, 
            category_names=category_names, 
            model_names=model_names, 
            postfix=postfix
        )

        # Save ensemble predictions 
        save_prediction_results(
            df_ensemble,
            "no_unknown",
            pred_result_folder_test,
            "-".join(sorted(model_names)),
            postfix
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ISIC-2019 Skin Lesion Classifier')
    parser.add_argument('data', metavar='DIR', help='path to data folder')
    parser.add_argument('--testfolder', metavar='DIR', help='path to test folder', default=None)
    parser.add_argument('--kfolds', help='Number of folds for k-fold cross validation (default: %(default)s)', default=0)
    parser.add_argument('--batchsize', type=int, help='Batch size (default: %(default)s)', default=32)
    parser.add_argument('--felr', type=float, help='Feature extractor learning rate (default: %(default)s)', default=1e-3)
    parser.add_argument('--ftlr', type=float, help='Fine tuning learning rate (default: %(default)s)', default=1e-5)
    parser.add_argument('--dropout', type=float, help='Dropout rate (default: %(default)s)')    
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
                            'Xception', 
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
                        default=["DenseNet201", "ResNet152"])
    parser.add_argument('--training', dest='training', action='store_true', help='Train models')
    parser.add_argument('--predtest', dest='predtest', action='store_true', help='Predict the test data')
    parser.add_argument('--online-data-augmentation-group', dest='online_dg_group', default=1, type=int)
    parser.add_argument('--unknown', nargs="*", type=float, default=0, help='Method to deal with unknown class')
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
    pred_result_folder_test = args.predtestresultfolder
    model_folder = args.modelfolder
    batch_size = args.batchsize
    max_queue_size = args.maxqueuesize
    felr = args.felr
    ftlr = args.ftlr    
    dropout = args.dropout
    fe_epochs = args.feepochs
    ft_epochs = args.ftepochs
    lmbda = args.l2 
    k_folds = int(args.kfolds)
    postfix = args.postfix
    online_dg_group = int(args.online_dg_group)
    unknown_method= args.unknown

    # ISIC data
    training_image_folder = os.path.join(data_folder, 'ISIC_2019_Training_Input')
    test_image_folder = args.testfolder if args.testfolder is not None else os.path.join(data_folder, 'ISIC_2019_Test_Input')

    # Ground truth
    ground_truth_file = os.path.join(data_folder, 'ISIC_2019_Training_GroundTruth.csv')
    df_ground_truth, known_category_names, unknown_category_name = load_isic_training_data(training_image_folder, ground_truth_file)
    class_weight_dict = compute_class_weight_dict(df_ground_truth)

    offline_dg_group = 1
    # METADATA File
    with open(os.path.join(data_folder, 'metadata.json')) as f:
        data_metadata = json.load(f)
        offline_dg_group = data_metadata["data_augmentation_group"]

    # Train models by Transfer Learning
    model_param_map = get_transfer_model_param_map() 

    base_model_params = [model_param_map[x] for x in args.models]

    if args.training:
        train(
            base_model_params, 
            df_ground_truth,  
            k_folds,
            len(known_category_names), 
            class_weight_dict, 
            batch_size, 
            max_queue_size, 
            ft_epochs, 
            model_folder,
            fe_epochs,
            dropout,
            felr,
            ftlr,
            lmbda,
            offline_dg_group,
            online_dg_group
        )

    # Predict Test Data
    if args.predtest:
        # Models used for predictition
        models_to_predict = []
        for base_model_param in base_model_params:
            models_to_predict.append(
                {
                    'model_name': base_model_param.class_name,
                    'input_size': base_model_param.input_size,
                    'preprocessing_function': base_model_param.preprocessing_func
                }
            )

        predidct_test(
            model_folder,
            test_image_folder,
            pred_result_folder_test,
            models_to_predict,
            known_category_names,
            unknown_method, 
            unknown_category_name,
            len(df_ground_truth['path'].tolist()),
            all(round(value, 2) == 1 for value in class_weight_dict.values()), 
            batch_size, 
            ft_epochs, 
            fe_epochs,
            dropout,
            felr,
            ftlr,
            lmbda,
            postfix,
            offline_dg_group,
            online_dg_group
        )