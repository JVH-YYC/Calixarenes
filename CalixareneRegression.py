"""
Plotting regressions for calixarene data
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import linregress

def load_single_regression_pickle(pickle_file_directory,
                                      pickle_file_name):
    """
    Docstring
    """

    pickle_file = pickle_file_directory + pickle_file_name
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    return data

def combine_regression_pickles(list_of_pickle_directories,
                                   list_of_pickle_file_names,
                                   target_split):
    """
    Docstring
    """

    combined_data = []
    for i in range(len(list_of_pickle_directories)):
        data = load_single_regression_pickle(list_of_pickle_directories[i],
                                                 list_of_pickle_file_names[i])
        working_data = data[target_split]
        combined_data = combined_data + working_data
    
    return combined_data

def create_dict_of_regression_pickles(list_of_list_of_pickle_directories,
                                          list_of_list_of_pickle_file_names,
                                          list_of_plot_labels,
                                          target_split):
        """
        Docstring
        """
    
        combined_data = {}
        for i in range(len(list_of_list_of_pickle_directories)):
            data = combine_regression_pickles(list_of_list_of_pickle_directories[i],
                                                  list_of_list_of_pickle_file_names[i],
                                                  target_split)
            combined_data[list_of_plot_labels[i]] = data
        
        return combined_data

def create_dict_of_mixed_regression_pickles(list_of_list_of_pickle_directories,
                                          list_of_list_of_pickle_file_names,
                                          list_of_plot_labels,
                                          target_split):
    """
    Docstring
    """

    combined_data = {}
    for i in range(len(list_of_list_of_pickle_directories)):
        data = combine_regression_pickles(list_of_list_of_pickle_directories[i],
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

def created_predicted_actual_regression_plot(data_dict):
    # Create a scatter plot
    for trial, pairs in data_dict.items():
        # Unpack the pairs into two lists: predicted and actual
        predicted, actual = zip(*pairs)

        # Calculate R^2 using linear regression
        slope, intercept, r_value, p_value, std_err = linregress(predicted, actual)
        r_squared = r_value ** 2

        # Plot and add the trial name and R^2 value to the legend
        plt.scatter(predicted, actual, label=f'{trial} ({r_squared:.2f})', alpha=0.5)

    # Plot the x=y line
    x = np.linspace(0, max(max(max(predicted), max(actual)) for _, pairs in data_dict.items()), 100)
    plt.plot(x, x, '--', color='gray')

    # Add legend, labels, and title
    plt.legend()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Predicted vs Actual Scatter Plot with R² Values')

    # Show the plot
    plt.show()    
    return    

def plot_separate_strategy_test_train(training_data,
                                      test_data):
    # Ensure both dictionaries have the same keys
    assert training_data.keys() == test_data.keys()

    for trial in training_data.keys():
        plt.figure()  # Create a new figure for each trial

        # Plot training data
        predicted, actual = zip(*training_data[trial])
        slope, intercept, r_value, _, _ = linregress(predicted, actual)
        r_squared_train = r_value ** 2
        plt.scatter(predicted, actual, label=f'{trial} (training) {r_squared_train:.2f}', color='blue')

        # Plot test data
        predicted, actual = zip(*test_data[trial])
        slope, intercept, r_value, _, _ = linregress(predicted, actual)
        r_squared_test = r_value ** 2
        plt.scatter(predicted, actual, label=f'{trial} (test) {r_squared_test:.2f}', color='green')

        # Plot the x=y line
        x = np.linspace(0, max(max(max(predicted), max(actual)) for _, pairs in test_data.items()), 100)
        plt.plot(x, x, '--', color='gray')

        # Add legend, labels, and title
        plt.legend()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Predicted vs Actual Scatter Plot for {trial}')

        # Show the plot
        plt.show()  
    return    

list_of_pd_1 = ['Raw Data/']*10
list_of_files_1 = ['CV0regression_svm_bypoint_each_test_train_data.pkl',
                 'CV1regression_svm_bypoint_each_test_train_data.pkl',
                 'CV2regression_svm_bypoint_each_test_train_data.pkl',
                 'CV3regression_svm_bypoint_each_test_train_data.pkl',
                 'CV4regression_svm_bypoint_each_test_train_data.pkl',
                 'CV5regression_svm_bypoint_each_test_train_data.pkl',
                 'CV6regression_svm_bypoint_each_test_train_data.pkl',
                 'CV7regression_svm_bypoint_each_test_train_data.pkl',
                 'CV8regression_svm_bypoint_each_test_train_data.pkl',
                 'CV9regression_svm_bypoint_each_test_train_data.pkl']

list_of_pd_2 = ['Raw Data/']*10
list_of_files_2 = ['CV0regression_svm_byhost_each_test_train_data.pkl',
                 'CV1regression_svm_byhost_each_test_train_data.pkl',
                 'CV2regression_svm_byhost_each_test_train_data.pkl',
                 'CV3regression_svm_byhost_each_test_train_data.pkl',
                 'CV4regression_svm_byhost_each_test_train_data.pkl',
                 'CV5regression_svm_byhost_each_test_train_data.pkl',
                 'CV6regression_svm_byhost_each_test_train_data.pkl',
                 'CV7regression_svm_byhost_each_test_train_data.pkl',
                 'CV8regression_svm_byhost_each_test_train_data.pkl',
                 'CV9regression_svm_byhost_each_test_train_data.pkl']

list_of_pd_3 = ['Raw Data/']*10
list_of_files_3 = ['CV0regression_svm_byhost_all_test_train_data.pkl',
                 'CV1regression_svm_byhost_all_test_train_data.pkl',
                 'CV2regression_svm_byhost_all_test_train_data.pkl',
                 'CV3regression_svm_byhost_all_test_train_data.pkl',
                 'CV4regression_svm_byhost_all_test_train_data.pkl',
                 'CV5regression_svm_byhost_all_test_train_data.pkl',
                 'CV6regression_svm_byhost_all_test_train_data.pkl',
                 'CV7regression_svm_byhost_all_test_train_data.pkl',
                 'CV8regression_svm_byhost_all_test_train_data.pkl',
                 'CV9regression_svm_byhost_all_test_train_data.pkl']

list_of_pd_4 = ['Raw Data/']*10
list_of_files_4 = ['CV0regression_rf_bypoint_each_test_train_data.pkl',
                 'CV1regression_rf_bypoint_each_test_train_data.pkl',
                 'CV2regression_rf_bypoint_each_test_train_data.pkl',
                 'CV3regression_rf_bypoint_each_test_train_data.pkl',
                 'CV4regression_rf_bypoint_each_test_train_data.pkl',
                 'CV5regression_rf_bypoint_each_test_train_data.pkl',
                 'CV6regression_rf_bypoint_each_test_train_data.pkl',
                 'CV7regression_rf_bypoint_each_test_train_data.pkl',
                 'CV8regression_rf_bypoint_each_test_train_data.pkl',
                 'CV9regression_rf_bypoint_each_test_train_data.pkl']

list_of_pd_5 = ['Raw Data/']*10
list_of_files_5 = ['CV0regression_rf_byhost_each_test_train_data.pkl',
                 'CV1regression_rf_byhost_each_test_train_data.pkl',
                 'CV2regression_rf_byhost_each_test_train_data.pkl',
                 'CV3regression_rf_byhost_each_test_train_data.pkl',
                 'CV4regression_rf_byhost_each_test_train_data.pkl',
                 'CV5regression_rf_byhost_each_test_train_data.pkl',
                 'CV6regression_rf_byhost_each_test_train_data.pkl',
                 'CV7regression_rf_byhost_each_test_train_data.pkl',
                 'CV8regression_rf_byhost_each_test_train_data.pkl',
                 'CV9regression_rf_byhost_each_test_train_data.pkl']

list_of_pd_6 = ['Raw Data/']*10
list_of_files_6 = ['CV0regression_rf_byhost_all_test_train_data.pkl',
                 'CV1regression_rf_byhost_all_test_train_data.pkl',
                 'CV2regression_rf_byhost_all_test_train_data.pkl',
                 'CV3regression_rf_byhost_all_test_train_data.pkl',
                 'CV4regression_rf_byhost_all_test_train_data.pkl',
                 'CV5regression_rf_byhost_all_test_train_data.pkl',
                 'CV6regression_rf_byhost_all_test_train_data.pkl',
                 'CV7regression_rf_byhost_all_test_train_data.pkl',
                 'CV8regression_rf_byhost_all_test_train_data.pkl',
                 'CV9regression_rf_byhost_all_test_train_data.pkl']

data_dict = create_dict_of_mixed_regression_pickles([list_of_pd_4, list_of_pd_5, list_of_pd_6],
                                                  [list_of_files_4, list_of_files_5, list_of_files_6],
                                                  ['By Point', 'By Host', 'By Host All'],
                                                  'test')

train_dict = create_dict_of_mixed_regression_pickles([list_of_pd_4, list_of_pd_5, list_of_pd_6],
                                                  [list_of_files_4, list_of_files_5, list_of_files_6],
                                                  ['By Point', 'By Host', 'By Host All'],
                                                  'train')
