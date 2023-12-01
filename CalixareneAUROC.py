"""
Calixarene project: file for plotting AUROC curves from pickled classification prediction data

Structure of pickled data is this:
Each pickle file is dictionary with 'train' and 'test' data.
In each of those, a list of tuples, where each tuple is (target, prediction)

This can be for categorization (target is 0 or 1, prediction is probability of 1) or regression.
"""

import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, auc
import numpy as np

def load_single_classification_pickle(pickle_file_directory,
                                      pickle_file_name):
    """
    Docstring
    """

    pickle_file = pickle_file_directory + pickle_file_name
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    return data

def combine_classification_pickles(list_of_pickle_directories,
                                   list_of_pickle_file_names,
                                   target_split):
    """
    Docstring
    """

    combined_data = []
    for i in range(len(list_of_pickle_directories)):
        data = load_single_classification_pickle(list_of_pickle_directories[i],
                                                 list_of_pickle_file_names[i])
        working_data = data[target_split]
        combined_data = combined_data + working_data
    
    return combined_data

def create_dict_of_classification_pickles(list_of_list_of_pickle_directories,
                                          list_of_list_of_pickle_file_names,
                                          list_of_plot_labels,
                                          target_split):
        """
        Docstring
        """
    
        combined_data = {}
        for i in range(len(list_of_list_of_pickle_directories)):
            data = combine_classification_pickles(list_of_list_of_pickle_directories[i],
                                                  list_of_list_of_pickle_file_names[i],
                                                  target_split)
            combined_data[list_of_plot_labels[i]] = data
        
        return combined_data

def create_dict_of_mixed_classification_pickles(list_of_list_of_pickle_directories,
                                          list_of_list_of_pickle_file_names,
                                          list_of_plot_labels,
                                          target_split):
    """
    Docstring
    """

    combined_data = {}
    for i in range(len(list_of_list_of_pickle_directories)):
        data = combine_classification_pickles(list_of_list_of_pickle_directories[i],
                                              list_of_list_of_pickle_file_names[i],
                                              target_split)

        # Check if the data needs reshaping
        if is_multi_target_format(data):
            data = reshape_multi_target_data(data)

        combined_data[list_of_plot_labels[i]] = data
    
    return combined_data

def is_multi_target_format(data):
    """
    Check if the given data is in multi-target format.

    Parameters:
    data (list): Data to check.

    Returns:
    bool: True if data is in multi-target format, False otherwise.
    """
    # Check if the first element is a tuple of numpy arrays with length 8
    return isinstance(data[0], tuple) and \
           len(data[0]) == 2 and \
           all(isinstance(elem, np.ndarray) and elem.shape[0] == 8 for elem in data[0])

def reshape_multi_target_data(data):
    """
    Reshape data from multi-target tuples to a dictionary format compatible with the plotting function.

    Parameters:
    data (list of tuples): Each tuple contains two numpy arrays (true labels, predicted probabilities)
    
    Returns:
    dict: Dictionary with keys as target labels and values as lists of tuples (class label, probability)
    """
def reshape_multi_target_data(data):
    """
    Reshape data from multi-target tuples to a single list of tuples format.

    Parameters:
    data (list of tuples): Each tuple contains two 1-dimensional numpy arrays (true labels, predicted probabilities)
    
    Returns:
    list: List of tuples (class label, probability)
    """
    reshaped_data = []

    for true_labels, pred_probs in data:
        # Check if the arrays are 1-dimensional and of the same length
        if true_labels.ndim != 1 or pred_probs.ndim != 1 or len(true_labels) != len(pred_probs):
            raise ValueError("Each array in the tuple must be 1-dimensional and of the same length")

        # Iterate over each element in the arrays and accumulate into a single list
        reshaped_data.extend(zip(true_labels, pred_probs))

    return reshaped_data

def plot_auc_roc(data):
    """
    Plot the AUROC curve from a list of tuples containing class labels and probabilities.

    Parameters:
    data (list of tuples): Each tuple is (class label, probability of class 1)
    """
    # Separate the class labels and probabilities
    y_true, y_scores = zip(*data)

    # Compute the ROC curve and AUROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plotting
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return

def plot_multiple_auc_roc(data_dict):
    """
    Plot multiple AUROC curves from a dictionary. Each key is the legend label, and 
    each value is a list of tuples containing class labels and probabilities.

    Parameters:
    data_dict (dict): Dictionary with keys as legend labels and values as lists of tuples (class label, probability)
    """
    plt.figure()

    for label, data in data_dict.items():
        # Separate the class labels and probabilities
        y_true, y_scores = zip(*data)

        # Compute the ROC curve and AUROC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Plotting
        plt.plot(fpr, tpr, lw=2, label=f'{label} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


list_of_pd_1 = ['Raw Data/']*10
list_of_files_1 = ['CV0classification_svm_bypoint_each_test_train_data.pkl',
                 'CV1classification_svm_bypoint_each_test_train_data.pkl',
                 'CV2classification_svm_bypoint_each_test_train_data.pkl',
                 'CV3classification_svm_bypoint_each_test_train_data.pkl',
                 'CV4classification_svm_bypoint_each_test_train_data.pkl',
                 'CV5classification_svm_bypoint_each_test_train_data.pkl',
                 'CV6classification_svm_bypoint_each_test_train_data.pkl',
                 'CV7classification_svm_bypoint_each_test_train_data.pkl',
                 'CV8classification_svm_bypoint_each_test_train_data.pkl',
                 'CV9classification_svm_bypoint_each_test_train_data.pkl']

list_of_pd_2 = ['Raw Data/']*10
list_of_files_2 = ['CV0classification_svm_byhost_each_test_train_data.pkl',
                 'CV1classification_svm_byhost_each_test_train_data.pkl',
                 'CV2classification_svm_byhost_each_test_train_data.pkl',
                 'CV3classification_svm_byhost_each_test_train_data.pkl',
                 'CV4classification_svm_byhost_each_test_train_data.pkl',
                 'CV5classification_svm_byhost_each_test_train_data.pkl',
                 'CV6classification_svm_byhost_each_test_train_data.pkl',
                 'CV7classification_svm_byhost_each_test_train_data.pkl',
                 'CV8classification_svm_byhost_each_test_train_data.pkl',
                 'CV9classification_svm_byhost_each_test_train_data.pkl']

list_of_pd_3 = ['Raw Data/']*10
list_of_files_3 = ['CV0classification_svm_byhost_all_test_train_data.pkl',
                 'CV1classification_svm_byhost_all_test_train_data.pkl',
                 'CV2classification_svm_byhost_all_test_train_data.pkl',
                 'CV3classification_svm_byhost_all_test_train_data.pkl',
                 'CV4classification_svm_byhost_all_test_train_data.pkl',
                 'CV5classification_svm_byhost_all_test_train_data.pkl',
                 'CV6classification_svm_byhost_all_test_train_data.pkl',
                 'CV7classification_svm_byhost_all_test_train_data.pkl',
                 'CV8classification_svm_byhost_all_test_train_data.pkl',
                 'CV9classification_svm_byhost_all_test_train_data.pkl']

list_of_pd_4 = ['Raw Data/']*10
list_of_files_4 = ['CV0classification_rf_bypoint_each_test_train_data.pkl',
                 'CV1classification_rf_bypoint_each_test_train_data.pkl',
                 'CV2classification_rf_bypoint_each_test_train_data.pkl',
                 'CV3classification_rf_bypoint_each_test_train_data.pkl',
                 'CV4classification_rf_bypoint_each_test_train_data.pkl',
                 'CV5classification_rf_bypoint_each_test_train_data.pkl',
                 'CV6classification_rf_bypoint_each_test_train_data.pkl',
                 'CV7classification_rf_bypoint_each_test_train_data.pkl',
                 'CV8classification_rf_bypoint_each_test_train_data.pkl',
                 'CV9classification_rf_bypoint_each_test_train_data.pkl']

list_of_pd_5 = ['Raw Data/']*10
list_of_files_5 = ['CV0classification_rf_byhost_each_test_train_data.pkl',
                 'CV1classification_rf_byhost_each_test_train_data.pkl',
                 'CV2classification_rf_byhost_each_test_train_data.pkl',
                 'CV3classification_rf_byhost_each_test_train_data.pkl',
                 'CV4classification_rf_byhost_each_test_train_data.pkl',
                 'CV5classification_rf_byhost_each_test_train_data.pkl',
                 'CV6classification_rf_byhost_each_test_train_data.pkl',
                 'CV7classification_rf_byhost_each_test_train_data.pkl',
                 'CV8classification_rf_byhost_each_test_train_data.pkl',
                 'CV9classification_rf_byhost_each_test_train_data.pkl']

list_of_pd_6 = ['Raw Data/']*10
list_of_files_6 = ['CV0classification_rf_byhost_all_test_train_data.pkl',
                 'CV1classification_rf_byhost_all_test_train_data.pkl',
                 'CV2classification_rf_byhost_all_test_train_data.pkl',
                 'CV3classification_rf_byhost_all_test_train_data.pkl',
                 'CV4classification_rf_byhost_all_test_train_data.pkl',
                 'CV5classification_rf_byhost_all_test_train_data.pkl',
                 'CV6classification_rf_byhost_all_test_train_data.pkl',
                 'CV7classification_rf_byhost_all_test_train_data.pkl',
                 'CV8classification_rf_byhost_all_test_train_data.pkl',
                 'CV9classification_rf_byhost_all_test_train_data.pkl']


data_dict = create_dict_of_mixed_classification_pickles([list_of_pd_1, list_of_pd_2, list_of_pd_3],
                                                  [list_of_files_1, list_of_files_2, list_of_files_3],
                                                  ['By Point', 'By Host', 'By Host All'],
                                                  'test')

train_dict = create_dict_of_mixed_classification_pickles([list_of_pd_1, list_of_pd_2, list_of_pd_3],
                                                  [list_of_files_1, list_of_files_2, list_of_files_3],
                                                  ['By Point', 'By Host', 'By Host All'],
                                                  'train')
