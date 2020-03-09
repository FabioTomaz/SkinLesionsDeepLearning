import argparse
import os
import datetime
# Import layers to load model
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras import utils
from data.data import load_isic_training_data, load_isic_training_and_out_dist_data, train_validation_split, compute_class_weight_dict, get_dataframe_from_img_folder
from vanilla_classifier import VanillaClassifier
from transfer_learn_classifier import TransferLearnClassifier
from metrics import balanced_accuracy
from base_model_param import get_transfer_model_param_map
from lesion_classifier import LesionClassifier
from utils import ensemble_predictions, formated_hyperparameter_str

# Use it to choose which GPU to train on
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

def main():
    parser = argparse.ArgumentParser(description='ISIC-2019 Skin Lesion Classifier')
    parser.add_argument('data', metavar='DIR', help='path to data folder')
    parser.add_argument('--testfolder', metavar='DIR', help='path to test folder', default=None)
    parser.add_argument('--batchsize', type=int, help='Batch size (default: %(default)s)', default=32)
    parser.add_argument('--felr', type=float, help='Feature extractor learning rate (default: %(default)s)', default=1e-3)
    parser.add_argument('--ftlr', type=float, help='Fine tuning learning rate (default: %(default)s)', default=1e-5)
    parser.add_argument('--dropout', type=float, help='Dropout rate (default: %(default)s)')    
    parser.add_argument('--maxqueuesize', type=int, help='Maximum size for the generator queue (default: %(default)s)', default=10)
    parser.add_argument('--feepochs', type=int, help='Feature extractor epochs (default: %(default)s)', default=3)    
    parser.add_argument('--epoch', type=int, help='Number of epochs (default: %(default)s)', default=100)
    parser.add_argument('--model', 
                        dest='models', 
                        nargs='*', 
                        choices=[
                            'Vanilla', 
                            'DenseNet121',
                            'DenseNet169',
                            'DenseNet201', 
                            'Xception', 
                            'NASNetLarge', 
                            'InceptionResNetV2', 
                            'VGG16', 
                            'VGG19', 
                            "EfficientNetB0", 
                            "EfficientNetB1",
                            "EfficientNetB2",
                            "EfficientNetB3",
                            "EfficientNetB4",
                            "EfficientNetB5", 
                            "EfficientNetB6",
                            "ResNet50",
                            "ResNet101",
                            "ResNet152",
                        ], 
                        help='Models')
    parser.add_argument('--training', dest='training', action='store_true', help='Train models')
    parser.add_argument('--predval', dest='predval', action='store_true', help='Predict validation set')
    parser.add_argument('--predtest', dest='predtest', action='store_true', help='Predict the test data.')
    parser.add_argument('--unknown', nargs="*", type=float, default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], help='Threshold to predict unknown class with.')
    parser.add_argument(
        '--predvalresultfolder', 
        help='Name of the prediction result folder for validation set (default: %(default)s)', 
        default='val_predict_results'
    )
    parser.add_argument(
        '--predtestresultfolder', 
        help='Name of the prediction result folder for test data (default: %(default)s)', 
        default='test_predict_results'
    )
    parser.add_argument('--modelfolder', help='Name of the model folder (default: %(default)s)', default='models')
    args = parser.parse_args()

    # Write command to a file
    with open('Cmd_History.txt', 'a') as f:
        f.write("{}\t{}\n".format(str(datetime.datetime.utcnow()), str(args)))

    data_folder = args.data
    pred_result_folder_val = args.predvalresultfolder
    pred_result_folder_test = args.predtestresultfolder
    model_folder = args.modelfolder
    batch_size = args.batchsize
    max_queue_size = args.maxqueuesize
    felr = args.felr
    ftlr = args.ftlr    
    dropout = args.dropout
    fe_epochs = args.feepochs
    epoch_num = args.epoch
    lmbda = None

    # ISIC data
    training_image_folder = os.path.join(data_folder, 'ISIC_2019_Training_Input')
    test_image_folder = args.testfolder if args.testfolder is not None else os.path.join(data_folder, 'ISIC_2019_Test_Input')

    # Ground truth
    ground_truth_file = os.path.join(data_folder, 'ISIC_2019_Training_GroundTruth.csv')
    df_ground_truth, known_category_names, unknown_category_name = load_isic_training_data(training_image_folder, ground_truth_file)
    category_names = known_category_names
    df_train, df_val = train_validation_split(df_ground_truth)
    class_weight_dict, class_weights = compute_class_weight_dict(df_train)
    category_num = len(category_names)

    # Models used for predictition
    models_to_predict = []
    workers = os.cpu_count()

    # Train Vanilla CNN
    if args.models is not None and 'Vanilla' in args.models:
        input_size_vanilla = (224, 224)
        if args.training:
            train_vanilla(df_train, df_val, category_num, class_weight_dict, batch_size, max_queue_size, epoch_num, input_size_vanilla, model_folder)
        models_to_predict.append({'model_name': 'Vanilla',
                                  'input_size': input_size_vanilla,
                                  'preprocessing_function': VanillaClassifier.preprocess_input})
    
    transfer_models = args.models.copy()
    if 'Vanilla' in transfer_models:
        transfer_models.remove('Vanilla')
    
    # Train models by Transfer Learning
    if args.models is not None:
        model_param_map = get_transfer_model_param_map() 
        base_model_params = [model_param_map[x] for x in transfer_models]
        if args.training:
            train_transfer_learning(
                base_model_params, 
                df_train, 
                df_val, 
                category_num, 
                class_weight_dict, 
                batch_size, 
                max_queue_size, 
                epoch_num, 
                model_folder,
                fe_epochs,
                dropout,
                felr,
                ftlr
            )
        for base_model_param in base_model_params:
            models_to_predict.append({'model_name': base_model_param.class_name,
                                      'input_size': base_model_param.input_size,
                                      'preprocessing_function': base_model_param.preprocessing_func})

    # Predict validation set
    if args.predval:
        os.makedirs(pred_result_folder_val, exist_ok=True)
        # Save Ground Truth of validation set
        val_ground_truth_file_path = os.path.join(pred_result_folder_val, 'Validation_Set_GroundTruth.csv')
        df_val.drop(columns=['path', 'category']).to_csv(
            path_or_buf=val_ground_truth_file_path, 
            index=False
        )
        print("Save \"{}\"".format(val_ground_truth_file_path))

        hyperparameter_str = formated_hyperparameter_str(
            fe_epochs,
            felr,
            ftlr,
            lmbda,
            dropout,
            batch_size,
            len(df_val['path'].tolist()) + len(df_train['path'].tolist()),
            len(set(class_weights)) <= 1
        )

        postfixes = ['best_balanced_acc', 'best_loss', 'latest']
        for postfix in postfixes:
            for m in models_to_predict:
                model_filepath = os.path.join(
                    model_folder,
                    m["model_name"],
                    hyperparameter_str, 
                    "{}.hdf5".format(postfix)
                )
                if os.path.exists(model_filepath):
                    print("===== Predict validation set using \"{}_{}\" model =====".format(m['model_name'], postfix))
                    model = load_model(filepath=model_filepath, custom_objects={'balanced_accuracy': balanced_accuracy(category_num)})

                    df_softmax_dict = LesionClassifier.predict_dataframe(
                        model=model, 
                        df=df_val,
                        category_names=category_names,
                        unknown_category=unknown_category_name if args.unknown else None,
                        augmentation_pipeline=LesionClassifier.create_aug_pipeline_val(m['input_size']),
                        preprocessing_function=m['preprocessing_function'],
                        batch_size=batch_size,
                        workers=workers,
                        unknown_thresholds=args.unknown
                    )

                    # Save results (multiple thresholds and no threhold)
                    for unknown_thresh, df_softmax in df_softmax_dict.items():
                        pred_folder = os.path.join(
                            pred_result_folder_test, 
                            m["model_name"],
                            hyperparameter_str,
                            f"unknown_{unknown_thresh}" if unknown_thresh != 1.0 else "no_unknown",
                        )
                        if not os.path.exists(pred_folder):
                            os.makedirs(pred_folder)

                    df_softmax.to_csv(path_or_buf=os.path.join(pred_folder, f"{postfix}.csv"), index=False)

                    del model
                    K.clear_session()
                else:
                    print("\"{}\" doesn't exist".format(model_filepath))

    # Predict Test Data
    if args.predtest:
        os.makedirs(pred_result_folder_test, exist_ok=True)
        df_test = get_dataframe_from_img_folder(test_image_folder, has_path_col=True)
        df_test.drop(columns=['path']).to_csv(os.path.join(pred_result_folder_test, 'ISIC_2019_Test.csv'), index=False)
        
        hyperparameter_str = formated_hyperparameter_str(
            fe_epochs,
            felr,
            ftlr,
            lmbda,
            dropout,
            batch_size,
            len(df_val['path'].tolist()) + len(df_train['path'].tolist()),
            len(set(class_weights)) <= 1,
        )
        postfix = 'best_balanced_acc'
        for m in models_to_predict:
            model_filepath = os.path.join(
                model_folder, 
                m["model_name"],
                hyperparameter_str,
                "{}.hdf5".format(postfix)
            )
            if os.path.exists(model_filepath):
                print("===== Predict test data using \"{}_{}\" model =====".format(m['model_name'], postfix))
                
                model = load_model(
                    filepath=model_filepath, 
                    custom_objects={'balanced_accuracy': balanced_accuracy(category_num)}
                )

                df_softmax_dict = LesionClassifier.predict_dataframe(
                    model=model, 
                    df=df_test,
                    category_names=category_names,
                    unknown_category=unknown_category_name if args.unknown else None,
                    augmentation_pipeline=LesionClassifier.create_aug_pipeline_val(m['input_size']),
                    preprocessing_function=m['preprocessing_function'],
                    batch_size=batch_size,
                    workers=workers,
                    unknown_thresholds=args.unknown
                )
                    
                # Save results (multiple thresholds and no threhold)
                for unknown_thresh, df_softmax in df_softmax_dict.items():
                    pred_folder = os.path.join(
                        pred_result_folder_test, 
                        m["model_name"],
                        hyperparameter_str,
                        f"unknown_{unknown_thresh}" if unknown_thresh != 1.0 else "no_unknown",
                    )
                    if not os.path.exists(pred_folder):
                        os.makedirs(pred_folder)

                    df_softmax.to_csv(path_or_buf=os.path.join(pred_folder, f"{postfix}.csv"), index=False)

                del model
                K.clear_session()
            else:
                print("\"{}\" doesn't exist".format(model_filepath))

        if (len(transfer_models)>1):
            # Ensemble Models' Predictions on Test Data
            df_ensemble = ensemble_predictions(
                result_folder=pred_result_folder_test, 
                hyperparameter_str=hyperparameter_str, 
                category_names=category_names, 
                save_file=False,
                model_names=transfer_models, 
                postfixes=[postfix]
            ).drop(columns=['pred_category'])

            df_ensemble.to_csv(
                os.path.join(
                    pred_result_folder_test, 
                    f"Ensemble_{postfix}.csv"
                ), 
                index=False
            )


def train_vanilla(
    df_train, 
    df_val, 
    num_classes, 
    class_weight_dict, 
    batch_size, 
    max_queue_size, 
    epoch_num, 
    input_size, 
    model_folder
):
    workers = os.cpu_count()

    classifier = VanillaClassifier(
        model_folder=model_folder,
        input_size=input_size,
        image_data_format=K.image_data_format(),
        num_classes=num_classes,
        batch_size=batch_size,
        max_queue_size=max_queue_size,
        class_weight=class_weight_dict,
        metrics=[balanced_accuracy(num_classes), 'accuracy'],
        image_paths_train=df_train['path'].tolist(),
        categories_train=utils.to_categorical(df_train['category'], num_classes=num_classes),
        image_paths_val=df_val['path'].tolist(),
        categories_val=utils.to_categorical(df_val['category'], num_classes=num_classes)
    )
    classifier.model.summary()
    print('Begin to train Vanilla CNN')
    classifier.train(epoch_num=epoch_num, workers=workers)
    del classifier
    K.clear_session()


def train_transfer_learning(
    base_model_params, 
    df_train, 
    df_val, 
    num_classes, 
    class_weight_dict, 
    batch_size, 
    max_queue_size, 
    epoch_num, 
    model_folder,
    fe_epochs,
    dropout,
    felr,
    ftlr
):
    workers = os.cpu_count()

    for model_param in base_model_params:
        classifier = TransferLearnClassifier(
            model_folder=model_folder,
            base_model_param=model_param,
            fc_layers=[512], # e.g. [512]
            num_classes=num_classes,
            dropout=dropout, 
            feature_extract_epochs= fe_epochs,
            feature_extract_start_lr=felr, 
            fine_tuning_start_lr=ftlr,
            batch_size=batch_size,
            max_queue_size=max_queue_size,
            image_data_format=K.image_data_format(),
            metrics=[balanced_accuracy(num_classes), 'accuracy'],
            class_weight=class_weight_dict,
            image_paths_train=df_train['path'].tolist(),
            categories_train=utils.to_categorical(df_train['category'], num_classes=num_classes),
            image_paths_val=df_val['path'].tolist(),
            categories_val=utils.to_categorical(df_val['category'], num_classes=num_classes)
        )
        classifier.model.summary()
        print("Begin to train {}".format(model_param.class_name))
        classifier.train(epoch_num=epoch_num, workers=workers)
        del classifier
        K.clear_session()


if __name__ == '__main__':
    main()
