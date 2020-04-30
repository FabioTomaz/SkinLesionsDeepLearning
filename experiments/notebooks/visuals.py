import os
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from helpers import filter_models_info, get_log_metric, get_test_metric
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from matplotlib.ticker import MaxNLocator


def plot_complexity_graph(
    csv_file, 
    title=None, 
    figsize=(14, 10), 
    feature_extract_epochs=None,
    loss_min=0, 
    loss_max=2, 
    epoch_min=None, 
    epoch_max=90, 
    accuracy_min=0, 
    accuracy_max=1,
    lr_min=0,
    lr_max=0.001
):
    df = pd.read_csv(csv_file)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=figsize)
    fig.patch.set_facecolor('white')
    fig.suptitle(title, fontsize=14)

    ax1.plot(df['loss'], label='Training Loss')
    ax1.plot(df['val_loss'], label='Validation Loss')
    ax1.set(title='Training and Validation Loss', xlabel='', ylabel='Loss')
    ax1.set_xlim([epoch_min, epoch_max])
    ax1.set_ylim([loss_min, loss_max])
    ax1.legend()

    ax2.plot(df['balanced_accuracy'], label='Training Accuracy')
    ax2.plot(df['val_balanced_accuracy'], label='Validation Accuracy')
    ax2.set(title='Training and Validation Accuracy', xlabel='Epoch', ylabel='Balanced Accuracy')
    ax2.set_xlim([epoch_min, epoch_max])
    ax2.set_ylim([accuracy_min, accuracy_max])
    ax2.legend()

    ax3.plot(df['lr'], label='Learning Rate')
    ax3.set(title='Learning rate over epochs', xlabel='Epoch', ylabel='Learning rate')
    ax3.set_xlim([epoch_min, epoch_max])
    ax3.set_ylim([0, df['lr'].max()+0.00001])
    ax3.legend()

    if feature_extract_epochs is not None:
        ax1.axvline(feature_extract_epochs-1, color='green', label='Start Fine Tuning')
        ax2.axvline(feature_extract_epochs-1, color='green', label='Start Fine Tuning')
        ax3.axvline(feature_extract_epochs-1, color='green', label='Start Fine Tuning')
        ax1.legend()
        ax2.legend()
        ax3.legend()
    
    # tight_layout() only considers ticklabels, axis labels, and titles. Thus, other artists may be clipped and also may overlap.
    # [left, bottom, right, top]
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def plot_grouped_2bars(scalars, scalarlabels, xticklabels, title=None, xlabel=None, ylabel=None):
    x = np.arange(len(xticklabels))  # the label locations
    width = 0.35  # the width of the bars

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title)
    fig.patch.set_facecolor('white')
    rects1 = ax.bar(x - width/2, scalars[0], width, label=scalarlabels[0])
    rects2 = ax.bar(x + width/2, scalars[1], width, label=scalarlabels[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    autolabel(ax, rects1)
    autolabel(ax, rects2)
    fig.tight_layout()


def autolabel(ax, rects):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    # References
        https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, figsize=(8, 6)):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    # References
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set(title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    im, cbar = heatmap(cm, classes, classes, ax=ax, cmap=plt.cm.Blues, cbarlabel='', grid=False)
    texts = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    return fig


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", grid=True, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    # References
        https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    if grid:
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"], threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_prob_bars(img_title, img_path, labels, probs, topk=5, title=None, figsize=(10, 4)):
    fig, (ax1, ax2) = plt.subplots(figsize=figsize, ncols=2)
    fig.patch.set_facecolor('white')

    if title is not None:
        fig.suptitle(title)

    ax1.set_title(img_title)
    ax1.imshow(plt.imread(img_path))

    # Plot probabilities bar chart
    ax2.set_title("Top {0} probabilities".format(topk))
    ax2.barh(np.arange(topk), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(topk))
    ax2.set_yticklabels(labels, size='medium')
    ax2.yaxis.tick_right()
    ax2.set_xlim(0, 1.0)
    ax2.invert_yaxis()
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def plot_class_dist(category_names, count_per_category):
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')
    ax.set(xlabel='Category', ylabel='Number of Images')
    rects = plt.bar(category_names, [count_per_category[i] for i in range(len(category_names))])
    autolabel(ax, rects)
    return fig 


def plot_hyperparameter_over_epochs(
    models_info, 
    metric,
    metric_label,
    hyperparameter_compared,
    hyperparameter_compared_label,
    constant_parameters={},
    title="", 
    subtitle=False,
    figsize=(14, 6), 
    pretrainedmodel="DenseNet201",
    feature_extract_epochs=None,
    epoch_max=None, 
    y_min=None,
    y_max=None,
    y_scale="linear"
):
    models_info_list = filter_models_info(
        models_info, 
        models=[pretrainedmodel],
        parameters=constant_parameters
    )

    if(len(models_info_list)==0):
        return  

    fig, ax1 = plt.subplots(
        nrows=1, 
        ncols=1, 
        figsize=figsize
    )
    fig.patch.set_facecolor('white')
    fig.suptitle(title, fontsize=14)

    log_metrics=[]
    param_values=[]
    for model_info in models_info_list:
        log_metrics.append(pd.read_csv(model_info["log"])[metric])
        param_values.append(float(model_info["hyperparameters"][hyperparameter_compared] if model_info["hyperparameters"][hyperparameter_compared] != "None" else 0))
    
    param_values, log_metrics = zip(*sorted(zip(param_values, log_metrics)))

    for i in range(len(log_metrics)):
        label = (hyperparameter_compared_label + "=" + str(param_values[i]) + " ")
        ax1.plot(log_metrics[i], label=label)

    if subtitle is True:
        subtitle = ""
        for key, value in model_info["hyperparameters"].items():
            if key is not hyperparameter_compared:
                subtitle += key + "=" + value + ", " 

    ax1.set(title=subtitle, xlabel='Epoch', ylabel=metric_label)
    ax1.set_xlim([0, epoch_max])
    if y_min is not None:
        ax1.set_ylim(bottom =y_min)
    if y_max is not None:
        ax1.set_ylim(top=y_max)
    ax1.grid(True)
    ax1.legend()

    if feature_extract_epochs is not None:
        ax1.axvline(feature_extract_epochs-1, color='green', label='Start Fine Tuning')
        #ax1.text(feature_extract_epochs-1, -0.06, str(feature_extract_epochs-1))
    #    ax1.legend()
    
    ax1.set_xscale("linear")
    ax1.set_yscale(y_scale)


    return fig


def plot_model_comparisson(
    models_info,
    df_ground_truth,
    metrics, 
    metric_labels,
    models=[],
    constant_parameters={},
    title="", 
    figsize=(10, 5), 
    y_min=0,
    y_max=100,
    parameter=None,
    parameter_filter=None,
    xticklabelfunction=None
):
    # Use some kind of benchmark hyperparameters in order to compare the models
    if parameter_filter == None:
        models_info_list = filter_models_info(
            models_info, 
            models=models,
            parameters=constant_parameters
        )
    else:
        models_info_list = []
        for i in parameter_filter:
            constant_parameters[parameter] = i
            models_info_list = models_info_list + filter_models_info(
                models_info, 
                models=models,
                parameters=constant_parameters
            )

    xticklabels = []
    scalars0 = []
    scalars1 = []
    for model_info in models_info_list:
        if model_info["pred_test"] is not None:
            # read true a prediction categories from validation dataset
            df = pd.merge(
                pd.read_csv(os.path.join(model_info["pred_test"], "no_unknown", "best_balanced_acc.csv")),
                df_ground_truth, 
                on='image'
            )
            y_true = df['category']
            y_pred = df['pred_category']
                

            # compute metric and associate it with model
            if parameter is None:
                label = str(model_info["model"])
            else: 
                label = str(model_info["hyperparameters"][parameter])

            if xticklabelfunction is not None:
                label = xticklabelfunction(label)
            xticklabels.append(label)

            scalars0.append(round(metrics[0](y_true, y_pred)*100,2))
            scalars1.append(round(metrics[1](y_true, y_pred)*100,2))

    scalars0, scalars1, xticklabels = zip(*sorted(zip(scalars0, scalars1, xticklabels)))
        
    x = np.arange(len(xticklabels))  # the label locations
    width = 0.40  # the width of the bars

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    fig.patch.set_facecolor('white')

    rects0 = ax.bar(x - width/2, scalars0, width, label=metric_labels[0])
    rects1 = ax.bar(x + width/2, scalars1, width, label=metric_labels[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set(xlabel="Model", ylabel="Metric Score (%)")
    ax.set_ylim(bottom =y_min, top=y_max)
    ax.legend()
    autolabel(ax, rects0)
    autolabel(ax, rects1)
    fig.tight_layout()
    
    return fig


def plot_model_comparisson_balanced_acc(
    models_info,
    df_ground_truth,
    constant_parameters={},
    models=None,
    title="", 
    figsize=(10, 5), 
    y_min=0,
    y_max=100,
    parameter=None,
    xticklabelfunction=None
):
    # Use some kind of benchmark hyperparameters in order to compare the models
    models_info_list = filter_models_info(
        models_info, 
        models=models,
        parameters=constant_parameters
    )

    xticklabels = []
    scalars0 = []
    scalars1 = []
    scalars2 = []
    for model_info in models_info_list:
        if model_info["pred_test"] is not None:
            # read true a prediction categories from validation dataset
            df_pred = pd.read_csv(os.path.join(model_info["pred_test"], "no_unknown", "best_balanced_acc.csv"))

            # compute metric and associate it with model
            if parameter is None:
                label = str(model_info["model"])
            else: 
                label = str(model_info["hyperparameters"][parameter])

            if xticklabelfunction is not None:
                label = xticklabelfunction(label)
            xticklabels.append(label)

            scalars0.append(round(get_log_metric(model_info["log"], metric="balanced_accuracy")*100,2))
            scalars1.append(round(get_log_metric(model_info["log"], metric="val_balanced_accuracy")*100,2))
            scalars2.append(round(get_test_metric(df_pred, df_ground_truth, balanced_accuracy_score)*100,2))

    scalars2, scalars1, scalars0, xticklabels = zip(*sorted(zip(scalars2, scalars1, scalars0, xticklabels)))
        
    x = np.arange(len(xticklabels))  # the label locations
    width = 0.30  # the width of the bars

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    fig.patch.set_facecolor('white')

    rects0 = ax.bar(x - width, scalars0, width, label="Train")
    rects1 = ax.bar(x, scalars1, width, label="Validation")
    rects2 = ax.bar(x + width, scalars2, width, label="Test")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set(xlabel="", ylabel="Balanced Accuracy (%)")
    ax.set_ylim(bottom =y_min, top=y_max)
    ax.legend()
    autolabel(ax, rects0)
    autolabel(ax, rects1)
    autolabel(ax, rects2)
    fig.tight_layout()
    
    return fig



def plot_model_parameter_comparisson(
    models_info,
    model_parameters,
    parameter_label="", 
    metric_label="",
    title="", 
    figsize=(7, 5)
): 
    parameter = []
    scalars_train = []
    scalars_val = []
    labels = []
    for model_info in models_info:
        # compute metric and associate it with model
        parameter.append(model_parameters[model_info["model"]])
        scalars_train.append(round(get_log_metric(model_info["log"], metric="balanced_accuracy")*100,2))
        scalars_val.append(round(get_log_metric(model_info["log"])*100,2))
        labels.append(model_info["model"])
            
    fig, ax = plt.subplots(figsize=figsize)

    parameter, scalars_val, scalars_train, labels = zip(*sorted(zip(parameter, scalars_val, scalars_train, labels)))

    ax.plot(parameter, scalars_train, '.b-', label="Train")
    ax.plot(parameter, scalars_val, '.r-', label="Validation")
    ax.set_title(title)
    ax.set(xlabel=parameter_label, ylabel=metric_label)

    ax.legend()
    ax.grid(True)

    #ax.set_xscale("log")
    #ax.set_yscale("linear")

    for i in range(len(labels)):
        ax.annotate(labels[i], (parameter[i], scalars_val[i]+0.1))
        ax.annotate(labels[i], (parameter[i], scalars_train[i]+0.1))

    fig.tight_layout()

    return fig


def plot_hyperparameter_comparisson(
    models_info,
    hyperparameter,
    val_metric="val_balanced_accuracy",
    train_metric="balanced_accuracy",
    test_metric=None,
    parameter_label="", 
    metric_label="Balanced Accuracy(%)",
    title="", 
    figsize=(7, 5),
    x_scale="linear",
    df_ground_truth=None,
    bar_plot=False,
    xticklabelfunction=lambda parameter_list: parameter_list,
    x_int_ticks=False
): 
    parameter = []
    scalars_train = []
    scalars_val = []
    scalars_test = []
    for model_info in models_info:
        param = 0.0 if model_info["hyperparameters"][hyperparameter]=="None" else float(model_info["hyperparameters"][hyperparameter]) 
        # compute metric and associate it with model
        parameter.append(param)
            
        scalars_train.append(round(get_log_metric(model_info["log"], metric=train_metric)*100,2))
        scalars_val.append(round(get_log_metric(model_info["log"], metric=val_metric)*100,2))
        if df_ground_truth is not None and test_metric is not None:
            # read true a prediction categories from validation dataset
            df_pred = pd.read_csv(os.path.join(model_info["pred_test_0"], "no_unknown", "best_balanced_acc.csv"))
            scalars_test.append(round(get_test_metric(df_pred, df_ground_truth, test_metric)*100,2))   
        else:
            scalars_test.append(0)
        
    fig, ax = plt.subplots(figsize=figsize)

    parameter, scalars_val, scalars_train, scalars_test = zip(*sorted(zip(parameter, scalars_val, scalars_train, scalars_test)))
    parameter=xticklabelfunction(parameter)

    if bar_plot is True:
        width = 0.28  # the width of the bars
        x = np.arange(len(parameter))  # the label locations
        rects0 = ax.bar(x - width/ 2 if test_metric is None else x - width, scalars_train, width, label="Train")
        autolabel(ax, rects0)
        rects1 = ax.bar(x + width/ 2 if test_metric is None else x , scalars_val, width, label="Validation")
        autolabel(ax, rects1)
        if df_ground_truth is not None and test_metric is not None:
            rects2 = ax.bar(x + width, scalars_test, width, label="Test")
            autolabel(ax, rects2)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xticks(x)
        ax.set_xticklabels(parameter)
    else:
        ax.plot(parameter, scalars_train, '.b-', label="Train")
        ax.plot(parameter, scalars_val, '.r-', label="Validation")
        if df_ground_truth is not None and test_metric is not None:
            ax.plot(parameter, scalars_test, '.g-', label="Test")
        ax.set_xscale(x_scale)
        if x_int_ticks is True:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)
        ax.set_yscale("linear")
    
    ax.set_title(title)
    ax.set(xlabel=parameter_label, ylabel=metric_label)
    ax.legend()
    fig.tight_layout()

    return fig
