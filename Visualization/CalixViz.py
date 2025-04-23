"""
Visualization scripts for calixarene project.

All LOO data is held in dictionaries with the following structure: {'host': {'peptide 1': (result),
'peptide 2': (result), ...}, 'host 2': {'peptide 1': (result), 'peptide 2': (result), ...}, ...}

Functions for creating AUROC plots, violin plots, predicted vs. actual scatters, as well as
numerical accuracy summaries (MSE, R^2, etc.) are included.

There is also the issue of 'accurate' vs. 'useful'. If a prediction has the correct trend of
adsorption predictions, but is off by a systematic error, this would be represented by a
slope close to 1, but an intercept far from 0. The is 'useful', but not 'accurate' (accurate will
have both slope and interecept within a certain range)

For a final test (after LOO was completed), different amouts of test data were held out. These dictionaries are of different structure.
The top-level key is simply the repeat round (from 0-19 for 20 repeats). Within these, each host points to a list of tuples, where the
tuples are (predicted, actual) values. From these, one can calculate the same metrics as above.
"""
import importlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import statistics
import calix_visual_settings as CVS
import matplotlib.patches as patches
import pickle
import copy
import random
import os

from sklearn.metrics import roc_curve, auc, r2_score, mean_squared_error
from scipy.stats import pearsonr

importlib.reload(CVS)

def load_result_dict(result_folder,
                     result_filename):
    """
    A simple function that uses pickle to load the result dictionary from a file.
    Returns the dictionary
    """
    with open(result_folder + result_filename, 'rb') as f:
        result_dict = pickle.load(f)
    return result_dict

def organize_by_host(pickle_file_folder,
                     list_of_pickle_files):
    """
    A function that takes a list of pickle files, and organizes the results by host.

    Returns a dictionary with the following structure: {'host': [(predicted, actual), (predicted, actual), ...],}
    for all hosts
    """
    organized_dict = {}
    for file in list_of_pickle_files:
        organized_dict[file] = {}
        result_dict = load_result_dict(pickle_file_folder,
                                       file)
        for host in result_dict:
            if host not in organized_dict:
                organized_dict[file][host] = []
            for peptide in result_dict[host]:
                organized_dict[file][host].append(result_dict[host][peptide])
    return organized_dict

def organize_by_peptide(pickle_file_folder,
                        list_of_pickle_files):
    """
    A function that takes a list of pickle files, and organizes the results by peptide.

    Returns a dictionary with the following structure: {'peptide': [(predicted, actual), (predicted, actual), ...],}
    for all peptides
    """
    organized_dict = {}
    for file in list_of_pickle_files:
        organized_dict[file] = {}
        result_dict = load_result_dict(pickle_file_folder,
                                       file)
        for host in result_dict:
            for peptide in result_dict[host]:
                if peptide not in organized_dict[file]:
                    organized_dict[file][peptide] = []
                organized_dict[file][peptide].append(result_dict[host][peptide])
    return organized_dict

def get_plot_setting(setting_key,
                     file_name=None,
                     calix_name=None,
                     peptide_name=None,
                     file_setting_dict=None,
                     calix_setting_dict=None,
                     peptide_setting_dict=None,
                     default_setting_dict=None):
    """
    Retrieve the plot setting value by checking the dictionaries in the specific order of priority:
    file_setting_dict > calix_setting_dict > peptide_setting_dict.
    
    Args:
    - setting_key (str): The key to search for in the dictionaries.
    - file_setting_dict (dict or None): The dictionary for file-specific settings.
    - calix_setting_dict (dict or None): The dictionary for calix-specific settings.
    - peptide_setting_dict (dict or None): The dictionary for peptide-specific settings.
    - default: The default value to return if the key is not found in any dictionaries.
    
    Returns:
    - The value associated with the setting_key, or the default value if not found.
    """
    if file_setting_dict is not None and file_name is not None and setting_key in file_setting_dict[file_name]:
        return file_setting_dict[file_name][setting_key]
    
    # Check in the calix_setting_dict based on the first character of the setting_key
    if calix_setting_dict is not None and calix_name is not None and setting_key in calix_setting_dict[calix_name[0]]:
        return calix_setting_dict[calix_name[0]][setting_key]
    
    if peptide_setting_dict is not None and peptide_name is not None and setting_key in peptide_setting_dict[peptide_name]:
        return peptide_setting_dict[peptide_name][setting_key]
    
    return default_setting_dict[setting_key]

def calculate_metrics(predicted_values, actual_values):
    """
    Calculate Mean Squared Error (MSE) and R² for given predicted and actual values.

    Also, uses the home-made metric of 'adjusted R2' to give an idea of when a relative trend was captured,
    but there was a systematic error
    
    Args:
    - predicted_values (list): List of predicted values.
    - actual_values (list): List of actual values.
    
    Returns:
    - mse (float): Mean Squared Error.
    - r2 (float): R² score.
    """
    mse = mean_squared_error(actual_values, predicted_values)
    r2 = r2_score(actual_values, predicted_values)
    adjusted_r2, shift_amount = calculate_adjusted_r2(actual_values, predicted_values)
    return mse, r2, adjusted_r2, shift_amount

def calculate_adjusted_r2(predicted_values, actual_values):
    """
    Calculates the 'best' possible R2 if all predictions are adjusted by the mean systematic error
    Returns that R2 value, as well as the amout of shift that was applied
    """

    mean_adj_list = [a - b for a, b in zip(actual_values, predicted_values)]

    systematic_error = statistics.mean(mean_adj_list)

    adjusted_pred_values = [a + systematic_error for a in predicted_values]

    adjusted_r2 = r2_score(actual_values, adjusted_pred_values)

    return adjusted_r2, systematic_error     

def calculate_slope_intercept(predicted_values, actual_values):
    """
    Calculate the slope and intercept of a linear fit to the scatter plot data.
    
    Args:
    - predicted_values (list): List of predicted values.
    - actual_values (list): List of actual values.
    
    Returns:
    - slope (float): Slope of the linear fit.
    - intercept (float): Intercept of the linear fit.
    """
    # Perform linear regression (1st degree polynomial fit)
    slope, intercept = np.polyfit(predicted_values, actual_values, 1)
    return slope, intercept

def save_slope_intercept_to_file(slope, intercept, output_name):
    """
    Save the slope and intercept to a .txt file.
    
    Args:
    - slope (float): Slope of the linear fit.
    - intercept (float): Intercept of the linear fit.
    - output_name (str): The base name for the output file.
    
    The file will be saved as output_name + "_slope_intercept.txt".
    """
    with open(f"{output_name}_slope_intercept.txt", "w") as file:
        file.write(f"Slope: {slope}\n")
        file.write(f"Intercept: {intercept}\n")

def save_metrics_to_file(metrics_dict, output_name):
    """
    Save the metrics dictionary to a .txt file.
    
    Args:
    - metrics_dict (dict): Dictionary containing metrics to be saved.
    - output_name (str): The base name for the output file.
    
    The file will be saved as output_name + ".txt".
    """
    with open(f"{output_name}.txt", "w") as file:
        for key, value in metrics_dict.items():
            file.write(f"{key}: {value}\n")
    return

def calculate_and_save_all_metrics(organized_dict,
                                   organize_by,
                                   group_hosts,
                                   output_name):
    """
    Calculate and save metrics like MSE and R², organized by the specified criteria.
    
    Args:
    - organized_dict (dict): The dictionary containing the organized data.
    - organize_by (str): The criterion by which the data is organized (e.g., 'peptide', 'host').
    - single_plot (bool): Whether to generate metrics for a single plot or multiple plots.
    - output_name (str): The base name for the output file.
    - host (str or None): Host identifier (if applicable).
    """
    for filename, data_dict in organized_dict.items():
        output_name = f"{output_name}_{filename}_{organize_by}"
        metrics_dict = {}

        if organize_by == 'peptide':
            for peptide, data_list in data_dict.items():
                predicted_values = [item['predicted'][0] if isinstance(item['predicted'], list) else item['predicted'] for item in data_list]
                actual_values = [item['actual'][0] if isinstance(item['actual'], list) else item['actual'] for item in data_list]                
                # Calculate metrics
                mse, r2, adjusted_r2, systematic_error = calculate_metrics(predicted_values, actual_values)
                
                # Store the metrics
                metrics_dict[f"Peptide {peptide} MSE"] = mse
                metrics_dict[f"Peptide {peptide} R²"] = r2
                metrics_dict[f"Peptide {peptide} Adjusted R²"] = adjusted_r2
                metrics_dict[f"Peptide {peptide} Systematic Error"] = systematic_error
        
        elif organize_by == 'host':
            if group_hosts:
                grouped_data = {}
                for host, data_list in data_dict.items():
                    host_group = host[0]
                    if host_group not in grouped_data:
                        grouped_data[host_group] = []
                    grouped_data[host_group].extend(data_list)
                
                for host_group, data_list in grouped_data.items():
                    predicted_values = [item['predicted'][0] if isinstance(item['predicted'], list) else item['predicted'] for item in data_list]
                    actual_values = [item['actual'][0] if isinstance(item['actual'], list) else item['actual'] for item in data_list]                
                    
                    # Calculate metrics for the grouped hosts
                    mse, r2, adjusted_r2, systematic_error = calculate_metrics(predicted_values, actual_values)
                    
                    # Store the metrics
                    metrics_dict[f"Host Group {host_group} MSE"] = mse
                    metrics_dict[f"Host Group {host_group} R²"] = r2
                    metrics_dict[f"Host Group {host_group} Adjusted R²"] = adjusted_r2
                    metrics_dict[f"Host Group {host_group} Systematic Error"] = systematic_error
            else:
                for host, data_list in data_dict.items():
                    predicted_values = [item['predicted'][0] if isinstance(item['predicted'], list) else item['predicted'] for item in data_list]
                    actual_values = [item['actual'][0] if isinstance(item['actual'], list) else item['actual'] for item in data_list]                
                    
                    # Calculate metrics for individual hosts
                    mse, r2, adjusted_r2, systematic_error = calculate_metrics(predicted_values, actual_values)
                    
                    # Store the metrics
                    metrics_dict[f"Host {host} MSE"] = mse
                    metrics_dict[f"Host {host} R²"] = r2
                    metrics_dict[f"Host {host} Adjusted R²"] = adjusted_r2
                    metrics_dict[f"Host {host} Systematic Error"] = systematic_error

        # Save metrics and fit results to file
        save_metrics_to_file(metrics_dict, output_name)
    
    return

def simple_calc_metrics_for_LOO(organized_dict):
    """
    Assumes a simple results dictionary (as from SKLearnBenchmarks LOO)
    Loops through individual calixarenes and prints R2 and adj R2

    CNN saves as single-item lists, so need to extract the value from the list
    """

    for entry in organized_dict:
        print('For calixarene:', entry)
        curr_results = organized_dict[entry]
        curr_pred = []
        curr_act = []
        for peptide in curr_results:
            if type(curr_results[peptide]['predicted']) == list:
                curr_pred.append(curr_results[peptide]['predicted'][0])
                curr_act.append(curr_results[peptide]['actual'][0])
            else:
                curr_pred.append(curr_results[peptide]['predicted'])
                curr_act.append(curr_results[peptide]['actual'])
        mse, r2, adj_r2, shift_amount = calculate_metrics(curr_pred, curr_act)
        print('R2:', r2)
        print('Adj R2:', adj_r2)

    return

def overall_r2_from_dict(organized_dict):
    """
    Simple function that calculates the overall R2 and adj R2 from a dictionary of results.
    Unlike all other function *no* split between any calixarene types
    """

    all_pred = []
    all_act = []
    for entry in organized_dict:
        curr_results = organized_dict[entry]
        curr_pred = []
        curr_act = []
        for peptide in curr_results:
            if type(curr_results[peptide]['predicted']) == list:
                curr_pred.append(curr_results[peptide]['predicted'][0])
                curr_act.append(curr_results[peptide]['actual'][0])
            else:
                curr_pred.append(curr_results[peptide]['predicted'])
                curr_act.append(curr_results[peptide]['actual'])
        all_pred.extend(curr_pred)
        all_act.extend(curr_act)

    mse, r2, adj_r2, shift_amount = calculate_metrics(all_pred, all_act)
    return r2, adj_r2

def calculate_bothr2_by_holdout(result_folder,
                                dict_of_holdout_amounts):
    """
    A function that takes the results of several different levels of test split, and calculates both the
    absolute and relative r2 values for each.

    It also enumerates what fraction of the total absolute predictions are above 0.6r2, and if not above 0.6r2,
    whether the relative is above 0.6adjr2
    """

    # Load the results from the files
    results_dict = {}
    for holdout_amt in dict_of_holdout_amounts:
        all_r2 = []
        all_adj_r2 = []
        results_dict[holdout_amt] = {}
        curr_dict = load_result_dict(result_folder,
                                        dict_of_holdout_amounts[holdout_amt])
        for repeat_trial in curr_dict:
            for curr_calix in curr_dict[repeat_trial]:
                if curr_calix[0] in ['A', 'E', 'P']:
                    curr_results = curr_dict[repeat_trial][curr_calix]
                    curr_predict = [x[0][0] if isinstance(x[0], list) else x[0] for x in curr_results]
                    curr_actual = [x[1][0] if isinstance(x[1], list) else x[1] for x in curr_results]
                    mse, r2, adjusted_r2, shift_amount = calculate_metrics(curr_predict, curr_actual)
                    all_r2.append(r2)
                    all_adj_r2.append(adjusted_r2)
        
        abs_r2_success = sum([1 for x in all_r2 if x > 0.7]) / len(all_r2)
        adj_r2_success = sum([1 for x, y in zip(all_r2, all_adj_r2) if x <= 0.7 and y > 0.7]) / len(all_r2)
        results_dict[holdout_amt]['r2_median'] = statistics.median(all_r2)
        results_dict[holdout_amt]['r2_average'] = statistics.mean(all_r2)
        results_dict[holdout_amt]['r2_success'] = abs_r2_success
        results_dict[holdout_amt]['adj_r2_median'] = statistics.median(all_adj_r2)
        results_dict[holdout_amt]['adj_r2_average'] = statistics.mean(all_adj_r2)
        results_dict[holdout_amt]['adj_r2_success'] = adj_r2_success
        results_dict[holdout_amt]['summed_success'] = abs_r2_success + adj_r2_success
    
    return results_dict

def evaluate_test_split_size(pickle_file_folder,
                             leading_string,
                             following_string,
                             holdout_amounts):
    """
    Another related workflow that looks at series of different test-holdout splits.

    Each of these holdouts was repeated 20x, and pickled as a dictionary. Top-level keys are the repeat number (str(0) to str(19)),
    following by the (predictable) calixarene in the test set (e.g. 'AP8':), which points to a list of [(predicted, actual), (predicted, actual)]

    Use existing functions to measure R2 and adjusted R2. Report the median adj R2, as well as the number of calixarenes with absolute R2 > 0.7,
    and adjusted R2 >0.7 (i.e. 'A+U')

    CNN output saves as single-item lists, so need to extract the value from the list 
    """

    for curr_holdout in holdout_amounts:
        curr_pickle_name = leading_string + str(curr_holdout) + following_string
        curr_results = load_result_dict(pickle_file_folder,
                                        curr_pickle_name)
        all_absolute_r2 = []
        all_adjusted_r2 = []
        for repeat_trial in curr_results.keys():
            for curr_calix in curr_results[repeat_trial]:
                if curr_calix[0] in ['A', 'E', 'P']:
                    calix_results = curr_results[repeat_trial][curr_calix]
                    curr_predict = [x[0] if isinstance(x[0], float) else x[0][0] for x in calix_results]
                    curr_actual = [x[1] if isinstance(x[1], float) else x[1][0] for x in calix_results]
                    mse, r2, adjusted_r2, shift_amount = calculate_metrics(curr_predict, curr_actual)
                    all_absolute_r2.append(r2)
                    all_adjusted_r2.append(adjusted_r2)
        
        median_r2 = statistics.median(all_absolute_r2)
        median_adj_r2 = statistics.median(all_adjusted_r2)

        abs_r2_success = sum([1 for x in all_absolute_r2 if x > 0.7]) / len(all_absolute_r2)
        adj_r2_success = sum([1 for x, y in zip(all_absolute_r2, all_adjusted_r2) if x <= 0.7 and y > 0.7]) / len(all_absolute_r2)

        print('For holdout:', curr_holdout)
        print(f"Median R2: {median_r2}")
        print(f"Median Adjusted R2: {median_adj_r2}")
        print(f"Fraction of Calixarenes with R2 > 0.7: {abs_r2_success}")
        print(f"Fraction of Calixarenes with R2 <= 0.7 and Adjusted R2 > 0.7: {adj_r2_success}")

    return 

def multi_scatter_plot(pickle_file_folder,
                    list_of_pickle_files,
                    file_setting_dict,
                    calix_setting_dict,
                    peptide_setting_dict,
                    default_setting_dict,
                    plot_setting_dict,
                    organize_by,
                    single_plot=True,
                    output_name=None,
                    save_fig=False,
                    calculate_metrics=False,
                    group_hosts=False):
    """
    A function that takes a translation dictionary, which has information about which files to
    access, and how each of these files should appear on the plot (has information about scatter
    plot shape, color, and opacity for each file)
    
    Plot_setting_dict has information about the figure size, axis labels, font choices (default is DejaVu Sans)
    tick marks, and other plot settings.

    Organize_by is a string that tells the function how to organize the data. If organized by 'host', then
    data should be organized by the top-level dictionary key. If organized by 'peptide', then data should be
    organized by the sub-level dictionary key. If organized by 'none', then all points should be plotted together
    with no distinction.

    If single_plots is true, then each distinction (peptide or host) will be plotted separately. If false, then
    the distinction made in organize_by should be reflected in the *shape of the points*. Colors are set in the
    translation_dict.
    """

    # Loop through the translation dictionary and plot each file, respecting organize_by and single_plots
    # The translation dictionary contains the relationship between file name and what should appear in plot legend
    # Data is organized using organize_by_host() and organize_by_peptide() functions above
    if organize_by == 'host':
        organized_dict = organize_by_host(pickle_file_folder,
                                          list_of_pickle_files)
    elif organize_by == 'peptide':
        organized_dict = organize_by_peptide(pickle_file_folder,
                                             list_of_pickle_files)
    else:
        list_of_data_dicts = []
        organized_dict = {}
        for file in list_of_pickle_files:
            list_of_data_dicts.append(load_result_dict(pickle_file_folder,
                                                       file))
        organized_dict['all'] = []
        for data_dict in list_of_data_dicts:
            for host in data_dict:
                for peptide in data_dict[host]:
                    organized_dict['all'].append(data_dict[host][peptide])

    if organize_by == 'none':
            # Extract the data to plot
            predicted_values = [item['predicted'] for item in organized_dict['all']]
            actual_values = [item['actual'] for item in organized_dict['all']]

            # Extract the plot settings from translation_dict
            color = default_setting_dict['color']
            size = default_setting_dict['size']
            opacity = default_setting_dict['opacity']
            marker = default_setting_dict['marker']

            # Set up the figure
            plt.figure(figsize=(plot_setting_dict['fig_width'], plot_setting_dict['fig_height']))
            plt.xlabel(plot_setting_dict['x_label'], fontsize=plot_setting_dict['axis_font_size'])
            plt.ylabel(plot_setting_dict['y_label'], fontsize=plot_setting_dict['axis_font_size'])
            plt.xticks(fontsize=plot_setting_dict['tick_font_size'])
            plt.yticks(fontsize=plot_setting_dict['tick_font_size'])
            plt.title(plot_setting_dict['title'], fontsize=plot_setting_dict['title_font_size'])


            # Create the scatter plot
            plt.scatter(predicted_values, actual_values, color=color, s=size, alpha=opacity, marker=marker)

            # Add a diagonal line for reference (optional, comment out if not needed)
            plt.plot([min(predicted_values), max(predicted_values)], 
                    [min(predicted_values), max(predicted_values)], 
                    color='black', linestyle='--', linewidth=1)

            # Show or save the figure
            if save_fig:
                plt.savefig(output_name, bbox_inches='tight')
            else:
                plt.show()        
    
    elif organize_by == 'peptide':
        if calculate_metrics == True:
            calculate_and_save_all_metrics(organized_dict,
                                           organize_by,
                                           single_plot,
                                           output_name)
        if single_plot:
            all_predicted_values = []
            all_actual_values = []
            # Set up the figure
            plt.figure(figsize=(plot_setting_dict['fig_width'], plot_setting_dict['fig_height']))
            plt.xlabel(plot_setting_dict['x_label'], fontsize=plot_setting_dict['axis_font_size'])
            plt.ylabel(plot_setting_dict['y_label'], fontsize=plot_setting_dict['axis_font_size'])
            plt.xticks(fontsize=plot_setting_dict['tick_font_size'])
            plt.yticks(fontsize=plot_setting_dict['tick_font_size'])
            plt.title(plot_setting_dict['title'], fontsize=plot_setting_dict['title_font_size'])

        # Iterate over each file
        for file, data_dict in organized_dict.items():
            # Iterate over each peptide
            for peptide, data_list in data_dict.items():
                # Extract the predicted and actual values for this peptide
                predicted_values = [item['predicted'] for item in data_list]
                actual_values = [item['actual'] for item in data_list]

                # Extract the plot settings from translation_dict for this peptide
                color = get_plot_setting('color',
                                         file_name=file,
                                         peptide_name=peptide,
                                         file_setting_dict=file_setting_dict,
                                         peptide_setting_dict=peptide_setting_dict,
                                         default_setting_dict=default_setting_dict)
                size = get_plot_setting('size',
                                        file_name=file,
                                        peptide_name=peptide,
                                        file_setting_dict=file_setting_dict,
                                        peptide_setting_dict=peptide_setting_dict,
                                        default_setting_dict=default_setting_dict)
                opacity = get_plot_setting('opacity',
                                             file_name=file,
                                             peptide_name=peptide,
                                             file_setting_dict=file_setting_dict,
                                             peptide_setting_dict=peptide_setting_dict,
                                             default_setting_dict=default_setting_dict)
                marker = get_plot_setting('marker',
                                            file_name=file,
                                            peptide_name=peptide,
                                            file_setting_dict=file_setting_dict,
                                            peptide_setting_dict=peptide_setting_dict,
                                            default_setting_dict=default_setting_dict)

                if single_plot:
                    # If single_plot is True, plot all peptides on the same figure
                    plt.scatter(predicted_values, actual_values, color=color, s=size, alpha=opacity, marker=marker, label=peptide)

                    # Add a diagonal line for reference for the single plot
                    all_predicted_values = all_predicted_values + predicted_values
                    all_actual_values = all_actual_values + actual_values

                else:
                    # If single_plot is False, create a new figure for each peptide
                    plt.figure()
                    plt.scatter(predicted_values, actual_values, color=color, s=size, alpha=opacity, marker=marker)

                    # Add a diagonal line for reference (optional)
                    plt.plot([min(predicted_values), max(predicted_values)], 
                            [min(predicted_values), max(predicted_values)], 
                            color='black', linestyle='--', linewidth=1)

                    plt.title(f'Scatter Plot for {peptide}')
                    plt.xlabel('Predicted Values')
                    plt.ylabel('Actual Values')
                    plt.grid(True)

                    # Show or save the figure
                    if save_fig:
                        plt.savefig(f"{output_name}_{peptide}.png", bbox_inches='tight')
                    else:
                        plt.show()
        if single_plot:
            plt.plot([min(all_predicted_values), max(all_predicted_values)], 
                    [min(all_actual_values), max(all_actual_values)], 
                    color='black', linestyle='--', linewidth=1)

            plt.legend(title='Peptide')
            plt.xlabel('Predicted Values')
            plt.ylabel('Actual Values')
            plt.title('Scatter Plot for All Peptides')
            plt.grid(True)
            # Show or save the figure
            if save_fig:
                plt.savefig(output_name, bbox_inches='tight')
            else:
                plt.show()

    elif organize_by == 'host':
        if calculate_metrics == True:
            calculate_and_save_all_metrics(organized_dict,
                                           organize_by,
                                           single_plot,
                                           output_name)
        if single_plot:
            # Initialize a set to track labels that have already been added to the legend
            seen_labels = set()

            all_predicted_values = []
            all_actual_values = []
            # Set up the figure
            plt.figure(figsize=(plot_setting_dict['fig_width'], plot_setting_dict['fig_height']))
            plt.xlabel(plot_setting_dict['x_label'], fontsize=plot_setting_dict['axis_font_size'])
            plt.ylabel(plot_setting_dict['y_label'], fontsize=plot_setting_dict['axis_font_size'])
            plt.xticks(fontsize=plot_setting_dict['tick_font_size'])
            plt.yticks(fontsize=plot_setting_dict['tick_font_size'])
            plt.title(plot_setting_dict['title'], fontsize=plot_setting_dict['title_font_size'])

        # Iterate over each file
        for file, data_dict in organized_dict.items():
            # Iterate over each peptide
            for host, data_list in data_dict.items():
                # Extract the predicted and actual values for this peptide
                predicted_values = [item['predicted'] for item in data_list]
                actual_values = [item['actual'] for item in data_list]

                # Extract the plot settings from translation_dict for this peptide
                color = get_plot_setting('color',
                                         file_name=file,
                                         calix_name=host,
                                         file_setting_dict=file_setting_dict,
                                         calix_setting_dict=calix_setting_dict,
                                         default_setting_dict=default_setting_dict)
                size = get_plot_setting('size',
                                        file_name=file,
                                        calix_name=host,
                                        file_setting_dict=file_setting_dict,
                                        calix_setting_dict=calix_setting_dict,
                                        default_setting_dict=default_setting_dict)
                opacity = get_plot_setting('opacity',
                                             file_name=file,
                                             calix_name=host,
                                             file_setting_dict=file_setting_dict,
                                             calix_setting_dict=calix_setting_dict,
                                             default_setting_dict=default_setting_dict)
                marker = get_plot_setting('marker',
                                            file_name=file,
                                            calix_name=host,
                                            file_setting_dict=file_setting_dict,
                                            calix_setting_dict=calix_setting_dict,
                                            default_setting_dict=default_setting_dict)

                if single_plot:
                    # If single_plot is True, plot all calixarenes on the same figure
                    if host[0] not in seen_labels:
                        plt.scatter(predicted_values, actual_values, color=color, s=size, alpha=opacity, marker=marker, label=host[0])
                        seen_labels.add(host[0])
                    else:
                        # Plot without adding to the legend if the label has already been seen
                        plt.scatter(predicted_values, actual_values, color=color, s=size, alpha=opacity, marker=marker)

                    all_predicted_values = all_predicted_values + predicted_values
                    all_actual_values = all_actual_values + actual_values

                else:
                    # If single_plot is False, create a new figure for each peptide
                    plt.figure()
                    plt.scatter(predicted_values, actual_values, color=color, s=size, alpha=opacity, marker=marker)

                    # Add a diagonal line for reference (optional)
                    plt.plot([min(predicted_values), max(predicted_values)], 
                            [min(predicted_values), max(predicted_values)], 
                            color='black', linestyle='--', linewidth=1)

                    plt.title(f'Scatter Plot for {host}')
                    plt.xlabel('Predicted Values')
                    plt.ylabel('Actual Values')
                    plt.grid(True)

                    # Show or save the figure
                    if save_fig:
                        plt.savefig(f"{output_name}_{host}.png", bbox_inches='tight')
                    else:
                        plt.show()
        if single_plot:
            plt.plot([min(all_predicted_values), max(all_predicted_values)], 
                    [min(all_actual_values), max(all_actual_values)], 
                    color='black', linestyle='--', linewidth=1)

            plt.legend(title='Peptide')
            plt.xlabel('Predicted Values')
            plt.ylabel('Actual Values')
            plt.title('Scatter Plot for All Peptides')
            plt.grid(True)
            # Show or save the figure
            if save_fig:
                plt.savefig(output_name, bbox_inches='tight')
            else:
                plt.show()   
    return

def calix_heatmap_from_csv(csv_folder,
                           csv_file_name,
                           output_name,
                           output_file_type,
                           heatmap_dict):
    """
    Docstring
    """
    # Read in the csv file
    csv_file = os.path.join(csv_folder, csv_file_name)
    df = pd.read_csv(csv_file, index_col=0, header=0)
    
    if heatmap_dict['log_val'] == True:
        df = np.log10(df)

    # Create the heatmap
    cmap = sns.diverging_palette(20, 220, sep=5, n=100, as_cmap=True)
    
    vmin = np.log10(1e-3)
    vmax = np.log10(1e2)
    center = np.log10(0.5)

    fig, ax = plt.subplots(figsize=(heatmap_dict['plot_width'],
                                    heatmap_dict['plot_height']))
    
    sns.heatmap(df, cmap=cmap, center=center, vmin=vmin, vmax=vmax, ax=ax)
    plt.subplots_adjust(left=0.2)

    rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                         fill=False, edgecolor='black', linewidth=1)
    ax.add_patch(rect)

    # Change font to DejaVu Sans for the row labels
    for label in ax.get_yticklabels():
        label.set_fontname(heatmap_dict['font_type'])
        label.set_fontsize(heatmap_dict['y_font_size'])
        label.align = 'left'   
    
    # Change font to DejaVus Sans for the column labels
    for label in ax.get_xticklabels():
        label.set_fontname(heatmap_dict['font_type'])
        label.set_fontsize(heatmap_dict['x_font_size'])
    
    # Change font type and size for the colobar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=heatmap_dict['tick_font_size'])

    # Remove y-axis label
    ax.set_ylabel('')

    #Update tick positions if necessary
    if heatmap_dict['tick_override'] == True:
        cbar.set_ticks(heatmap_dict['tick_positions'])
        cbar.set_ticklabels(heatmap_dict['tick_labels'])


    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1)    
    
    # Save the heatmap
    plt.savefig(output_name,
                dpi=300,
                facecolor="white",
                bbox_inches='tight',
                pad_inches=0.05,
                format=output_file_type)
    plt.close()
    return

def scatter_by_network_class(pickle_file_folder,
                             pickle_file_dict,
                             calix_plot_setting,
                             output_name,
                             plot_mode='abs',
                             save_png=False):
    """
    Script to make predicted vs actual plots segregated by calixarene type and network.

    Further segregated into 'absolute' and 'relative' predictions - same definitions as elsewhere

    Network type determines point color; 'predictable' vs 'unpredictable determines' point type

    Only for LOO results. Dataset split trials only considered A/E/P calixarenes - we want to examine B/C/D here as well
    """

    # Load the results from the files
    results_dict = {}
    
    for specific_file in pickle_file_dict:
        results_dict[specific_file] = {}
        results_dict[specific_file]['pred'] = {}
        results_dict[specific_file]['pred']['predicted'] = []
        results_dict[specific_file]['pred']['actual'] = []
        results_dict[specific_file]['unpred'] = {}
        results_dict[specific_file]['unpred']['predicted'] = []
        results_dict[specific_file]['unpred']['actual'] = []

        curr_dict = load_result_dict(pickle_file_folder,
                                     pickle_file_dict[specific_file])
        # Need to gather points as intermediate lists, so that when output mode = 'rel', we adjust
        # the predicted/actual numbers by the mean error

        for curr_calix in curr_dict:
            curr_pred_list = []
            curr_act_list = []
            curr_results = curr_dict[curr_calix]
            if curr_calix[0] in ['A', 'E', 'P']:
                for curr_peptide in curr_results:    
                    curr_pred = curr_results[curr_peptide]['predicted']
                    curr_act = curr_results[curr_peptide]['actual']
                    if type(curr_pred) == list:
                        curr_pred = curr_pred[0]
                    if type(curr_act) == list:
                        curr_act = curr_act[0]
                    curr_pred_list.append(curr_pred)
                    curr_act_list.append(curr_act)
                if plot_mode == 'abs':
                    results_dict[specific_file]['pred']['predicted'].extend(curr_pred_list)
                    results_dict[specific_file]['pred']['actual'].extend(curr_act_list)
                else:
                    mean_adj_list = [a - b for a, b in zip(curr_act_list, curr_pred_list)]
                    systematic_error = statistics.mean(mean_adj_list)
                    adjusted_pred_values = [a + systematic_error for a in curr_pred_list]
                    results_dict[specific_file]['pred']['predicted'].extend(adjusted_pred_values)
                    results_dict[specific_file]['pred']['actual'].extend(curr_act_list)

            else:
                for curr_peptide in curr_results:    
                    curr_pred = curr_results[curr_peptide]['predicted']
                    curr_act = curr_results[curr_peptide]['actual']
                    if type(curr_pred) == list:
                        curr_pred = curr_pred[0]
                    if type(curr_act) == list:
                        curr_act = curr_act[0]
                    curr_pred_list.append(curr_pred)
                    curr_act_list.append(curr_act)
                if plot_mode == 'abs':
                    results_dict[specific_file]['unpred']['predicted'].extend(curr_pred_list)
                    results_dict[specific_file]['unpred']['actual'].extend(curr_act_list)
                else:
                    mean_adj_list = [a - b for a, b in zip(curr_act_list, curr_pred_list)]
                    systematic_error = statistics.mean(mean_adj_list)
                    adjusted_pred_values = [a + systematic_error for a in curr_pred_list]
                    results_dict[specific_file]['unpred']['predicted'].extend(adjusted_pred_values)
                    results_dict[specific_file]['unpred']['actual'].extend(curr_act_list)

    # Set up the figure
    plt.figure(figsize=(calix_plot_setting['fig_width'], calix_plot_setting['fig_height']))
    plt.xlabel(calix_plot_setting['x_label'], fontsize=calix_plot_setting['axis_font_size'])
    plt.ylabel(calix_plot_setting['y_label'], fontsize=calix_plot_setting['axis_font_size'])
    plt.xticks(fontsize=calix_plot_setting['tick_font_size'])
    plt.yticks(fontsize=calix_plot_setting['tick_font_size'])
    plt.title(calix_plot_setting['title'], fontsize=calix_plot_setting['title_font_size'])

    # Create the scatter plot, with different colors for 'predictable' and 'unpredictable', and
    # different shapes for the different files included (the names will be the dictionary keys)

    for specific_file in results_dict:
        predictable_pred_val = results_dict[specific_file]['pred']['predicted']
        predictable_act_val = results_dict[specific_file]['pred']['actual']
        unpredictable_pred_val = results_dict[specific_file]['unpred']['predicted']
        unpredictable_act_val = results_dict[specific_file]['unpred']['actual']
        # Extract the plot settings from calix_plot_setting
        pred_color = calix_plot_setting['scatter_color'][specific_file]['Predictable']
        pred_shape = calix_plot_setting['scatter_shape'][specific_file]['Predictable']
        unpred_color = calix_plot_setting['scatter_color'][specific_file]['Unpredictable']
        unpred_shape = calix_plot_setting['scatter_shape'][specific_file]['Unpredictable']
        size = calix_plot_setting['scatter_size']
        pred_opacity = calix_plot_setting['scatter_opacity'][specific_file]['Predictable']
        unpred_opacity = calix_plot_setting['scatter_opacity'][specific_file]['Unpredictable']
        # Plot the predictable points
        plt.scatter(predictable_pred_val,
                    predictable_act_val,
                    color=pred_color,
                    s=size,
                    alpha=pred_opacity,
                    marker=pred_shape,
                    label=specific_file + ' Predictable')
        # Try unpredicable; sometimes must be skipped for filtered datasets
        try:
            print('R2 value for "predictable" points is:', str(r2_score(predictable_act_val, predictable_pred_val)))
            # Plot the unpredictable points
            plt.scatter(unpredictable_pred_val,
                        unpredictable_act_val,
                        color=unpred_color,
                        s=size,
                        alpha=unpred_opacity,
                        marker=unpred_shape,
                        label=specific_file + ' Unpredictable')
            print('R2 value for "unpredictable" points is:', str(r2_score(unpredictable_act_val, unpredictable_pred_val)))
            full_act_val = predictable_act_val + unpredictable_act_val
            full_pred_val = predictable_pred_val + unpredictable_pred_val
            print('R2 value for full set is:', str(r2_score(full_act_val, full_pred_val)))
        except:
            print('No unpredictable points for this dataset')
            pass
    # Set the x and y axis to equal max/min to enforce a square plot,
    # and add a diagonal line and the legent in the top right

    # Read current max/min
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    # Set the limits to be equal at the max value. Print/output legend separately so it can be combined in Illustrator
    max_val = max(x_max, y_max)
    min_val = min(x_min, y_min)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1)
    
    # Get current legend info from main plot
    handles, labels = plt.gca().get_legend_handles_labels()

    if save_png:
        plt.savefig(output_name + '.png',
                    dpi=300,
                    facecolor="white",
                    bbox_inches='tight',
                    pad_inches=0.05,
                    format='png')

    plt.show()

    # Create empty figure just for legend
    fig_legend = plt.figure(figsize=(2, 1))  # tweak size as needed

    fig_legend.legend(handles=handles,
                    labels=labels,
                    loc='center',
                    frameon=False,  # No box around legend
                    fontsize=calix_plot_setting['legend_font_size'])  # Optional: use your setting

    fig_legend.gca().axis('off')

    if save_png:
        fig_legend.savefig(output_name + 'legend_only.png',
                        bbox_inches='tight',
                        transparent=True)

    return

def scatter_holdout_amount(pickle_file_folder,
                           pickle_file_dict,
                           calix_plot_setting,
                           output_name,
                           plot_mode='abs',
                           save_png=False):
    
    """ 
    A function very closely related to that directly above - with only 1 real difference.
    This function interacts with the 20-repeat test split dictionaries, which have a different
    dictionary organization, so the initial data wrangling is slightly different.
    Plotting occurs in the same way once the data is rectified
    """
    # Helper function for adjusting pred/act values on the fly
    def adjusted_r2_raw(pred_list, act_list):
        """
        Adjust predicted values by average error and return
        """
        mean_adj_list = [a - b for a, b in zip(act_list, pred_list)]

        systematic_error = statistics.mean(mean_adj_list)

        adjusted_pred_values = [a + systematic_error for a in pred_list]

        return adjusted_pred_values
    
    # Compile into single result dict
    results_dict = {}

    for specific_file in pickle_file_dict:
        results_dict[specific_file] = {}
        results_dict[specific_file]['predicted'] = []
        results_dict[specific_file]['actual'] = []
        results_dict[specific_file]['adjusted'] = []

        curr_dict = load_result_dict(pickle_file_folder,
                                     pickle_file_dict[specific_file])
        
        for repeat in curr_dict:
            for calix in curr_dict[repeat]:
                curr_pred = [x[0][0] if isinstance(x[0], list) else x[0] for x in curr_dict[repeat][calix]]
                curr_act = [x[1][0] if isinstance(x[1], list) else x[1] for x in curr_dict[repeat][calix]]
                curr_adj = adjusted_r2_raw(curr_pred, curr_act)
                
                results_dict[specific_file]['predicted'].extend(curr_pred)
                results_dict[specific_file]['actual'].extend(curr_act)
                results_dict[specific_file]['adjusted'].extend(curr_adj)

    # Set up the figure
    plt.figure(figsize=(calix_plot_setting['fig_width'], calix_plot_setting['fig_height']))
    plt.xlabel(calix_plot_setting['x_label'], fontsize=calix_plot_setting['axis_font_size'])
    plt.ylabel(calix_plot_setting['y_label'], fontsize=calix_plot_setting['axis_font_size'])
    plt.xticks(fontsize=calix_plot_setting['tick_font_size'])
    plt.yticks(fontsize=calix_plot_setting['tick_font_size'])
    plt.title(calix_plot_setting['title'], fontsize=calix_plot_setting['title_font_size'])

    # Create the plot, which might be multiple network types, multiple holdout amounts, etc.
    for specific_file in results_dict:
        if plot_mode == 'abs':
            current_pred_val = results_dict[specific_file]['predicted']
            current_act_val = results_dict[specific_file]['actual']
        else:
            current_pred_val = results_dict[specific_file]['adjusted']
            current_act_val = results_dict[specific_file]['actual']
        # Extract the plot settings from calix_plot_setting
        color = calix_plot_setting['scatter_color'][specific_file]
        shape = calix_plot_setting['scatter_shape'][specific_file]
        size = calix_plot_setting['scatter_size']
        opacity = calix_plot_setting['scatter_opacity'][specific_file] # Different splits have different numbers of points
        # Plot the current points
        plt.scatter(current_pred_val,
                    current_act_val,
                    color=color,
                    s=size,
                    alpha=opacity,
                    marker=shape,
                    label=specific_file)
    # Set the x and y axis to equal max/min to enforce a square plot,
    # add a diagonal line. Plot legend separately so it can be combined in Illustrator

    # Read current max/min
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    # Set the limits to be equal at the max value. Print/output legend separately so it can be combined in Illustrator
    max_val = max(x_max, y_max)
    min_val = min(x_min, y_min)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1)

    # Get current legend info from main plot
    handles, labels = plt.gca().get_legend_handles_labels()

    if save_png:
        plt.savefig(output_name + '.png',
                    dpi=300,
                    facecolor="white",
                    bbox_inches='tight',
                    pad_inches=0.05,
                    format='png')

    plt.show()

    # Create empty figure just for legend
    fig_legend = plt.figure(figsize=(2, 1))  # tweak size as needed

    fig_legend.legend(handles=handles,
                    labels=labels,
                    loc='center',
                    frameon=False,  # No box around legend
                    fontsize=calix_plot_setting['legend_font_size'])  # Optional: use your setting

    fig_legend.gca().axis('off')

    if save_png:
        fig_legend.savefig(output_name + 'legend_only.png',
                        bbox_inches='tight',
                        transparent=True)

    return

def highlight_individual_scatter(pickle_file_folder,
                                 pickle_file_name,
                                 highlight_calix,
                                 calix_plot_setting,
                                 output_name,
                                 plot_mode='abs',
                                 save_png=False):
    """
    Script to make a scatter plot where a small number of specific calixarenes are emphasized against the backdrop of
    all other calixarenes.

    Only does so for 1 network - so no pickle_file_dict, just a list of calix.
    """

    # Load the results from the files
    results_dict = {}
    results_dict['All Others'] = {}
    results_dict['All Others']['predicted'] = []
    results_dict['All Others']['actual'] = []

    for highlight in highlight_calix:
        results_dict[highlight] = {}
        results_dict[highlight]['predicted'] = []
        results_dict[highlight]['actual'] = []

    open_result_dict = load_result_dict(pickle_file_folder,
                                        pickle_file_name)

    for curr_calix in open_result_dict:
        curr_pred_list = []
        curr_act_list = []
        curr_results = open_result_dict[curr_calix]

        for curr_peptide in curr_results:    
            curr_pred = curr_results[curr_peptide]['predicted']
            curr_act = curr_results[curr_peptide]['actual']
            if type(curr_pred) == list:
                curr_pred = curr_pred[0]
            if type(curr_act) == list:
                curr_act = curr_act[0]
            curr_pred_list.append(curr_pred)
            curr_act_list.append(curr_act)
        if plot_mode == 'abs':
            update_pred_list = curr_pred_list
        else:
            mean_adj_list = [a - b for a, b in zip(curr_act_list, curr_pred_list)]
            systematic_error = statistics.mean(mean_adj_list)
            update_pred_list = [a + systematic_error for a in curr_pred_list]

        if curr_calix in highlight_calix:
            results_dict[curr_calix]['predicted'].extend(update_pred_list)
            results_dict[curr_calix]['actual'].extend(curr_act_list)
        else:
            results_dict['All Others']['predicted'].extend(update_pred_list)
            results_dict['All Others']['actual'].extend(curr_act_list)

    # Set up the figure
    plt.figure(figsize=(calix_plot_setting['fig_width'], calix_plot_setting['fig_height']))
    plt.xlabel(calix_plot_setting['x_label'], fontsize=calix_plot_setting['axis_font_size'])
    plt.ylabel(calix_plot_setting['y_label'], fontsize=calix_plot_setting['axis_font_size'])
    plt.xticks(fontsize=calix_plot_setting['tick_font_size'])
    plt.yticks(fontsize=calix_plot_setting['tick_font_size'])
    plt.title(calix_plot_setting['title'], fontsize=calix_plot_setting['title_font_size'])

    # Create the scatter plot, with different colors for each highlight and "All Others"
    

    for calix in results_dict:
        current_pred_val = results_dict[calix]['predicted']
        current_act_val = results_dict[calix]['actual']

        # Extract the plot settings from calix_plot_setting
        color = calix_plot_setting['scatter_color'][calix]
        shape = calix_plot_setting['scatter_shape'][calix]
        size = calix_plot_setting['scatter_size']
        opacity = calix_plot_setting['scatter_opacity'][calix]
        # Plot the current points
        plt.scatter(current_pred_val,
                    current_act_val,
                    color=color,
                    s=size,
                    alpha=opacity,
                    marker=shape,
                    label=calix)
        
    # Set the x and y axis to equal max/min to enforce a square plot,
    # and add a diagonal line and the legent in the top right

    # Read current max/min
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    # Set the limits to be equal at the max value. Print/output legend separately so it can be combined in Illustrator
    max_val = max(x_max, y_max)
    min_val = min(x_min, y_min)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1)
    
    # Get current legend info from main plot
    handles, labels = plt.gca().get_legend_handles_labels()

    if save_png:
        plt.savefig(output_name + '.png',
                    dpi=300,
                    facecolor="white",
                    bbox_inches='tight',
                    pad_inches=0.05,
                    format='png')

    plt.show()

    # Create empty figure just for legend
    fig_legend = plt.figure(figsize=(2, 1))  # tweak size as needed

    fig_legend.legend(handles=handles,
                    labels=labels,
                    loc='center',
                    frameon=False,  # No box around legend
                    fontsize=calix_plot_setting['legend_font_size'])  # Optional: use your setting

    fig_legend.gca().axis('off')

    if save_png:
        fig_legend.savefig(output_name + 'legend_only.png',
                        bbox_inches='tight',
                        transparent=True)

    return

def calculate_test_calix_distribution(pickle_file_folder,
                                      network_name_list,
                                      calixarene_name_list,
                                      holdout_amount_list,
                                      leading_string,
                                      repeat_number):
    """
    A function that takes a folder full of different training/test splits, and looks at the distribution of
    specific calixarenes in each trial. It will export a dictionary with the maximum and minimum number of times a calixarene
    is observed in a given trial, and a True/False flag for whether any calixarene was never in test set
    """

    # Initialize the dictionary to hold the results
    return_dict = {}
    blank_dict = {}
    for entry in calixarene_name_list:
        return_dict[entry] = [100, 0, False] #[min, max, never in test set]
        blank_dict[entry] = [0, False]
    # Loop through the network names
    for network_name in network_name_list:
        for holdout_amount in holdout_amount_list:
            file_name_abs = f"{leading_string} {holdout_amount} HO {network_name} absolute.pkl"
            file_name_rel = f"{leading_string} {holdout_amount} HO {network_name} relative.pkl"
            print('Loading files:', file_name_abs, file_name_rel)
            # Load the absolute and relative files
            abs_dict = load_result_dict(pickle_file_folder,
                                        file_name_abs)
            abs_dict = rectify_pno2_name(abs_dict)
            rel_dict = load_result_dict(pickle_file_folder,
                                        file_name_rel)
            rel_dict = rectify_pno2_name(rel_dict)

            # Create a working copy of the blank starting dict
            abs_working_dict = copy.deepcopy(blank_dict)
            rel_working_dict = copy.deepcopy(blank_dict)
            # Loop through both dictionaries, which have identical keys
            for repeat in abs_dict:
                for calix in abs_dict[repeat]:
                    abs_working_dict[calix][0] += 1
                for calix in rel_dict[repeat]:
                    rel_working_dict[calix][0] += 1

            # Now check if any calixarene was never in the test set
            for calix in abs_working_dict:
                if abs_working_dict[calix][0] == 0:
                    abs_working_dict[calix][1] = True
            for calix in rel_working_dict:
                if rel_working_dict[calix][0] == 0:
                    rel_working_dict[calix][1] = True

            # Update the return dictionary with the max value, and any 'True' flags
            for calix in return_dict:
                return_dict[calix][0] = min(return_dict[calix][0],
                                            abs_working_dict[calix][0],
                                            rel_working_dict[calix][0])
                return_dict[calix][1] = max(return_dict[calix][1],
                                            abs_working_dict[calix][0],
                                            rel_working_dict[calix][0])
                if abs_working_dict[calix][1] or rel_working_dict[calix][1] == True:
                    return_dict[calix][2] = True
    
    return return_dict

def rectify_pno2_name(input_dict):
    """
    A trivial function that takes an input dictionary.
    If 'P-NO2' is a key in the dictionary, it is replaced with
    'PNO2', and the fixed dictionary is returned. Due to a typo between
    different training .csv sheets

    This mainly impacts the 20-repeat holdout dictionaries, so must
    loop through repeat trials.

    Also, some dicts use the string '0', versus the int 0. Rectify all to strings.
    """

    # Create a new dictionary to hold the fixed keys
    fixed_dict = {}

    # Iterate through the input dictionary
    for repeat in input_dict:
        for key, value in input_dict[repeat].items():
            # Check if the key is 'P-NO2' and replace it with 'PNO2'
            if key == 'P-NO2':
                fixed_key = 'PNO2'
            else:
                fixed_key = key

            # Add the value to the new dictionary with the fixed key
            if str(repeat) not in fixed_dict:
                fixed_dict[str(repeat)] = {}
            fixed_dict[str(repeat)][fixed_key] = value

    return fixed_dict

def normalize_and_report_test_splits(pickle_file_folder,
                                     network_name_list,
                                     calixarene_name_list,
                                     holdout_amount_list,
                                     leading_string,
                                     repeat_number,
                                     output_name):
    """
    A function that takes many different test splits, for many different models. First, it calculates
    overall distributions for the entire set.

    Then, it begins to parse individual splits and networks, normalizing each example with the correct
    number of repeats per calixarene.

    Most calix have between 15-18 repeats, so multiple copies will be needed. In the case of odd numbers of
    test examples, take the middle value. In the case of even, take the 2 closest to the middle, and repeat them
    equal numbers of times (up until the final example for an odd set, which is randomly chosen.)

    Adjusted R2 values need to be calculated for each calixarene - otherwise conflicting systematic errors will
    cancel out, and the adjustment won't work

    Exclude any calix that did not appear in a given model/split trial.
    """

    # Helper function for extracting the middle of a list
    def middle_slice(lst, k):
        """
        Return a sublist of length k taken from the middle of lst.
        If (len(lst) - k) is odd, randomly choose between the two center positions.
        """
        n = len(lst)
        # if k >= n just return a shallow copy of the whole list
        if k >= n:
            return lst[:]

        rem = n - k               # number to drop
        base = rem // 2           # floor of the start index
        offset = random.randint(0, rem % 2)
        start = base + offset
        return lst[start : start + k]
    
    # Helper function for adjusting pred/act values on the fly
    def adjusted_r2_raw(pred_list, act_list):
        """
        Adjust predicted values by average error and return
        """
        mean_adj_list = [a - b for a, b in zip(act_list, pred_list)]

        systematic_error = statistics.mean(mean_adj_list)

        adjusted_pred_values = [a + systematic_error for a in pred_list]

        return adjusted_pred_values
      
    # Calculate overall stats
    stat_dict = calculate_test_calix_distribution(pickle_file_folder,
                                                    network_name_list,
                                                    calixarene_name_list,
                                                    holdout_amount_list,
                                                    leading_string,
                                                    repeat_number)
    # Create a dictionary for final reporting
    report_dict = {}

    # Loop through the network type, then holdout amount.
    for network_name in network_name_list:
        if network_name not in report_dict:
            report_dict[network_name] = {}
        for holdout_amount in holdout_amount_list:
            if holdout_amount not in report_dict[network_name]:
                report_dict[network_name][holdout_amount] = {}
                report_dict[network_name][holdout_amount]['abs'] = {}
                report_dict[network_name][holdout_amount]['rel'] = {}

                # Load absolute and relative files
                file_name_abs = f"{leading_string} {holdout_amount} HO {network_name} absolute.pkl"
                file_name_rel = f"{leading_string} {holdout_amount} HO {network_name} relative.pkl"
                abs_dict = load_result_dict(pickle_file_folder,
                                            file_name_abs)
                abs_dict = rectify_pno2_name(abs_dict)
                rel_dict = load_result_dict(pickle_file_folder,
                                            file_name_rel)
                rel_dict = rectify_pno2_name(rel_dict)

                # Process each calixarene
                for calix in calixarene_name_list:
                    if stat_dict[calix][2] == True:
                        # If the calixarene was never in the test set, skip it
                        continue
                    rel_r2s = []
                    abs_r2s = []
                    for repeat in abs_dict:
                        if calix in abs_dict[repeat]:
                            # Get the predicted and actual values
                            curr_pred = [x[0][0] if isinstance(x[0], list) else x[0] for x in abs_dict[repeat][calix]]
                            curr_act = [x[1][0] if isinstance(x[1], list) else x[1] for x in abs_dict[repeat][calix]]

                            # Calculate the R2 value
                            r2_val = r2_score(curr_act, curr_pred)
                            abs_r2s.append([r2_val, curr_act, curr_pred, adjusted_r2_raw(curr_pred, curr_act)])
                        if calix in rel_dict[repeat]:
                            # Get the predicted and actual values
                            curr_pred = [x[0][0] if isinstance(x[0], list) else x[0] for x in rel_dict[repeat][calix]]
                            curr_act = [x[1][0] if isinstance(x[1], list) else x[1] for x in rel_dict[repeat][calix]]

                            # Calculate the R2 value
                            r2_val = r2_score(curr_act, curr_pred)
                            rel_r2s.append([r2_val, curr_act, curr_pred, adjusted_r2_raw(curr_pred, curr_act)])
                    
                    # Now, we need to normalize the R2 values by the number of repeats from the stat_dict
                    abs_pred_list = []
                    abs_adj_list = []
                    abs_act_list = []
                    rel_pred_list = []
                    rel_act_list = []
                    rel_adj_list = []

                    # First, organize lists by R2 value
                    abs_r2s.sort(key=lambda x: x[0], reverse=True)
                    rel_r2s.sort(key=lambda x: x[0], reverse=True)
                    # Get the number of necessary repeats for this calixarene and figure out how many times to repeat all entries
                    num_obs = stat_dict[calix][1]
                    abs_repeats = num_obs // len(abs_r2s)
                    rel_repeats = num_obs // len(rel_r2s)
                    abs_remainder = num_obs % len(abs_r2s)
                    rel_remainder = num_obs % len(rel_r2s)
                    abs_slice = middle_slice(abs_r2s, num_obs)
                    rel_slice = middle_slice(rel_r2s, num_obs)

                    # Fill the lists with the correct number of repeats
                    for entry in abs_r2s:
                        for repeat in range(abs_repeats):
                            abs_pred_list.extend(entry[2])
                            abs_act_list.extend(entry[1])
                            abs_adj_list.extend(entry[3])
                    for entry in rel_r2s:
                        for repeat in range(rel_repeats):
                            rel_pred_list.extend(entry[2])
                            rel_act_list.extend(entry[1])
                            rel_adj_list.extend(entry[3])

                    # Add the remainder entries to the end of the list
                    for entry in abs_slice:
                        abs_pred_list.extend(entry[2])
                        abs_act_list.extend(entry[1])
                        abs_adj_list.extend(entry[3])
                    for entry in rel_slice:
                        rel_pred_list.extend(entry[2])
                        rel_act_list.extend(entry[1])
                        rel_adj_list.extend(entry[3])

                    # Save to the return dictionary in separate pred/actual lists - easier for further processing
                    report_dict[network_name][holdout_amount]['abs'][calix] = {}
                    report_dict[network_name][holdout_amount]['abs'][calix]['predicted'] = abs_pred_list
                    report_dict[network_name][holdout_amount]['abs'][calix]['actual'] = abs_act_list
                    report_dict[network_name][holdout_amount]['abs'][calix]['adjusted'] = abs_adj_list
                    report_dict[network_name][holdout_amount]['rel'][calix] = {}
                    report_dict[network_name][holdout_amount]['rel'][calix]['predicted'] = rel_pred_list
                    report_dict[network_name][holdout_amount]['rel'][calix]['actual'] = rel_act_list
                    report_dict[network_name][holdout_amount]['rel'][calix]['adjusted'] = rel_adj_list

    # Save the report dictionary to a pickle file
    pickle_file_name = os.path.join(pickle_file_folder,
                                    output_name + '.pkl')
    with open(pickle_file_name, 'wb') as f:
        pickle.dump(report_dict, f)

    return report_dict
                        
def report_various_test_split_results(test_split_dict):
    """"
    Simple function that reads the dictionary generated directly above, and reports the R2 and adjusted R2 for
    transfer to an Excel sheet and/or other reporting
    """                                             

    # Depends on standard dictionary structure
    for network_type in test_split_dict:
        print('For network type:', network_type)
        for holdout_amount in test_split_dict[network_type]:
            print('For holdout amount:', holdout_amount)
            abs_predictable_pred = []
            abs_predictable_act = []
            abs_predictable_adj = []
            rel_predictable_pred = []
            rel_predictable_act = []
            rel_predictable_adj = []

            for calix in test_split_dict[network_type][holdout_amount]['abs']:
                abs_predictable_pred.extend(test_split_dict[network_type][holdout_amount]['abs'][calix]['predicted'])
                abs_predictable_act.extend(test_split_dict[network_type][holdout_amount]['abs'][calix]['actual'])
                abs_predictable_adj.extend(test_split_dict[network_type][holdout_amount]['abs'][calix]['adjusted'])
            for calix in test_split_dict[network_type][holdout_amount]['rel']:
                rel_predictable_pred.extend(test_split_dict[network_type][holdout_amount]['rel'][calix]['predicted'])
                rel_predictable_act.extend(test_split_dict[network_type][holdout_amount]['rel'][calix]['actual'])
                rel_predictable_adj.extend(test_split_dict[network_type][holdout_amount]['rel'][calix]['adjusted'])

            # Calculate the R2 values
            abs_predictable_r2 = r2_score(abs_predictable_act, abs_predictable_pred)
            abs_predictable_adj_r2 = r2_score(abs_predictable_act, abs_predictable_adj)
            print('Absolute Predictable R2:', abs_predictable_r2)
            print('Absolute Predictable Adjusted R2:', abs_predictable_adj_r2)
            rel_predictable_r2 = r2_score(rel_predictable_act, rel_predictable_pred)
            rel_predictable_adj_r2 = r2_score(rel_predictable_act, rel_predictable_adj)
            print('Relative Predictable R2:', rel_predictable_r2)
            print('Relative Predictable Adjusted R2:', rel_predictable_adj_r2)
    
    return

def line_plot_various_test_split(test_split_results_dict,
                                 calix_plot_setting,
                                 networks_to_plot,
                                 output_name,
                                 y_range=[0, 1],
                                 plot_mode='abs',
                                 save_png=False):
    """
    An alternative way of visualizing various test splits: line plots.

    Results dictionaries are explicitly defined below, levels of organization in dicts
    is network type --> absolute or relative training --> holdout amount --> raw or adjusted r2
    """

    # Set up the figure
    plt.figure(figsize=(calix_plot_setting['fig_width'], calix_plot_setting['fig_height']))
    plt.xlabel(calix_plot_setting['x_label'], fontsize=calix_plot_setting['axis_font_size'])
    plt.ylabel(calix_plot_setting['y_label'], fontsize=calix_plot_setting['axis_font_size'])
    plt.xticks(fontsize=calix_plot_setting['tick_font_size'])
    plt.yticks(fontsize=calix_plot_setting['tick_font_size'])
    plt.title(calix_plot_setting['title'], fontsize=calix_plot_setting['title_font_size'])

    # Change tick direction if desired
    plt.tick_params(axis='both',
                    which='both',
                    direction='in')

    # Create the line plot, with line type and marker type set in the plot_setting dict
    for network_type in networks_to_plot:
        x_pos = []
        abs_r2s = []
        rel_r2s = []
        for holdout_amount in test_split_results_dict[network_type]['absolute']:
            x_pos.append(holdout_amount * 100)
            if plot_mode == 'abs':
                abs_r2s.append(test_split_results_dict[network_type]['absolute'][holdout_amount]['raw'])
                rel_r2s.append(test_split_results_dict[network_type]['relative'][holdout_amount]['raw'])
            else:
                abs_r2s.append(test_split_results_dict[network_type]['absolute'][holdout_amount]['adj'])
                rel_r2s.append(test_split_results_dict[network_type]['relative'][holdout_amount]['adj'])

            # Extract the plot settings from calix_plot_setting
            color = calix_plot_setting['marker_color'][network_type]
            abs_shape = calix_plot_setting['marker_shape']['abs']
            rel_shape = calix_plot_setting['marker_shape']['rel']
            size = calix_plot_setting['marker_size']
            opacity = calix_plot_setting['marker_opacity']
            abs_line = calix_plot_setting['line_style']['abs']
            rel_line = calix_plot_setting['line_style']['rel']
        
        # Adjust y-axis limits
        plt.ylim(y_range[0], y_range[1])
        # Plot the current points
        plt.plot(x_pos,
                    abs_r2s,
                    color=color,
                    marker=abs_shape,
                    markersize=size,
                    alpha=opacity,
                    linestyle=abs_line,
                    linewidth=2)
        plt.plot(x_pos,
                    rel_r2s,
                    color=color,
                    marker=rel_shape,
                    markersize=size,
                    alpha=opacity,
                    linestyle=rel_line,
                    linewidth=2)

    # Get current legend info from main plot
    handles, labels = plt.gca().get_legend_handles_labels()

    if save_png:
        plt.savefig(output_name + '.png',
                    dpi=300,
                    facecolor="white",
                    bbox_inches='tight',
                    pad_inches=0.05,
                    format='png')

    plt.show()

    # Create empty figure just for legend
    fig_legend = plt.figure(figsize=(2, 1))  # tweak size as needed

    fig_legend.legend(handles=handles,
                    labels=labels,
                    loc='center',
                    frameon=False,  # No box around legend
                    fontsize=calix_plot_setting['legend_font_size'])  # Optional: use your setting

    fig_legend.gca().axis('off')

    if save_png:
        fig_legend.savefig(output_name + 'legend_only.png',
                        bbox_inches='tight',
                        transparent=True)
        
    return

calix_plot_setting = {'fig_width': 8,
                        'fig_height': 8,
                        'x_label': 'Predicted (relative)',
                        'y_label': 'Actual (relative)',
                        'axis_font_size': 32,
                        'tick_font_size': 24,
                        'title_font_size': 40,
                        'legend_font_size': 32,
                        'title': 'AFP Abs vs Relative',
                        'scatter_color': {'AFP ABS': {'Predictable': (0.055, 0.297, 0.344),
                                                      'Unpredictable': (0.055, 0.297, 0.344)},
                                          'AFP REL': {'Predictable': (0.73, 0.29, 0.15),
                                                        'Unpredictable': (0.73, 0.29, 0.15)}},
                        'scatter_shape': {'AFP ABS': {'Predictable': 'o',
                                                      'Unpredictable': 'o'},
                                          'AFP REL': {'Predictable': 'D',
                                                      'Unpredictable': 'D'}},
                        'scatter_size': 75,
                        'scatter_opacity': {'AFP ABS': {'Predictable': 0.4,
                                                        'Unpredictable': 0.4},
                                            'AFP REL': {'Predictable': 0.6,
                                                        'Unpredictable': 0.6}}}

lead_in_plot_setting = {'fig_width': 8,
                        'fig_height': 8,
                        'x_label': 'Predicted',
                        'y_label': 'Actual',
                        'axis_font_size': 32,
                        'tick_font_size': 24,
                        'title_font_size': 40,
                        'legend_font_size': 32,
                        'title': 'AFP',
                        'scatter_color': {'AM1': (0.727, 0.285, 0.152),
                                          'AH5': (0.98, 0.69, 0.25), 
                                        'E11': (0.398, 0.176, 0.566),
                                        'All Others': (0.055, 0.297, 0.344)},
                        'scatter_shape': {'AM1': 'o',
                                          'AH5': 'o',
                                          'E11': 'o',
                                          'All Others': 'o'},
                        'scatter_size': 75,
                        'scatter_opacity': {'AM1': 0.95,
                                            'AH5': 0.95,
                                            'E11': 0.95,
                                            'All Others': 0.5}}

lead_in_calix_list = []

holdout_plot_setting = {'fig_width': 8,
                        'fig_height': 8,
                        'x_label': 'Predicted',
                        'y_label': 'Actual',
                        'axis_font_size': 32,
                        'tick_font_size': 24,
                        'title_font_size': 40,
                        'legend_font_size': 32,
                        'title': 'AFP',
                        'scatter_color': {'0.25': (0.727, 0.285, 0.152),
                                          '0.50': (0.000, 0.578, 0.266),
                                        '0.75': (0.398, 0.176, 0.566)},
                        'scatter_shape': {'0.25': 'o',
                                          '0.50': 'o',
                                          '0.75': 'o'},
                        'scatter_size': 25,
                        'scatter_opacity': {'0.25': 0.5,
                                            '0.50': 0.3,
                                            '0.75': 0.1}}

holdout_file_dict = {'0.25': '20 split 0.25 HO AFP relative.pkl',
                     '0.50': '20 split 0.5 HO AFP relative.pkl',
                     '0.75': '20 split 0.75 HO AFP relative.pkl'}

holdout_line_setting = {'fig_width': 8,
                        'fig_height': 2.5,
                        'x_label': 'Holdout Amount (%)',
                        'y_label': 'R2',
                        'axis_font_size': 32,
                        'tick_font_size': 24,
                        'title_font_size': 40,
                        'legend_font_size': 32,
                        'title': '',
                        'marker_color': {'RF': (0.727, 0.285, 0.152),
                                          'CNN': (0.000, 0.578, 0.266),
                                        'AFP': (0.398, 0.176, 0.566)},
                        'marker_shape': {'rel': 'D',
                                         'abs': 'o'},
                        'marker_size': 15,
                        'marker_opacity': 0.95,
                        'line_style': {'rel': ':',
                                       'abs': '--'}}

line_plot_include_list = ['RF', 'CNN', 'AFP']

highlight_plot_setting = {'fig_width': 8,
                        'fig_height': 8,
                        'x_label': 'Predicted',
                        'y_label': 'Actual',
                        'axis_font_size': 32,
                        'tick_font_size': 24,
                        'title_font_size': 40,
                        'legend_font_size': 32,
                        'title': 'GCN',
                        'scatter_color': {'AO3': (0.727, 0.285, 0.152), 
                                        'AM1': (0.000, 0.578, 0.266),
                                        'CP2': (0.398, 0.176, 0.566),
                                        'All Others': (0.055, 0.297, 0.344)},
                        'scatter_shape': {'AO3': 'o',
                                          'AM1': 'o',
                                          'CP2': 'o',
                                          'All Others': 'o'},
                        'scatter_size': 75,
                        'scatter_opacity': {'AO3': 0.95,
                                            'AM1': 0.95,
                                            'CP2': 0.95,
                                            'All Others': 0.1}}

highlight_calix_list = ['AO3', 'AM1', 'CP2']

example_plot_dict = {'AttentiveFP': 'AttentiveFP_regression.pkl'}
pickle_file_dict = {'Absolute': 'AttentiveFP_regression.pkl',
                    'Relative': 'Relative_FP.pkl'}

full_calix_list = ['AP1', 'AP3', 'AP4', 'AP5', 'AP6', 'AP7', 'AP8',
                   'AP9', 'AH1', 'AH2', 'AH3', 'AH4', 'AH5', 'AH6',
                   'AH7', 'AM1', 'AM2', 'AO1', 'AO2', 'AO3', 'E1',
                   'E3', 'E6', 'E7', 'E8', 'E11', 'PNO2', 'PSC4',
                   'BP0', 'BP1', 'BH2', 'BM1', 'CP1', 'CP2', 'DP2',
                   'DM1', 'DO2', 'DO3']

network_name_list = ['RF', 'SVM', 'CNN', 'GCN', 'AFP']
holdout_amount_list = [0.1, 0.15, 0.25, 0.5, 0.75]
leading_string = '20 split'

rf_abs_dict = {'0.05': '20 split 0.05 HO RF absolute.pkl',
               '0.1': '20 split 0.1 HO RF absolute.pkl',
               '0.15': '20 split 0.15 HO RF absolute.pkl',
               '0.25': '20 split 0.25 HO RF absolute.pkl',
               '0.5': '20 split 0.5 HO RF absolute.pkl',
               '0.75': '20 split 0.75 HO RF absolute.pkl'}

rf_rel_dict = {'0.05': '20 split 0.05 HO RF relative.pkl',
                '0.1': '20 split 0.1 HO RF relative.pkl',
                '0.15': '20 split 0.15 HO RF relative.pkl',
                '0.25': '20 split 0.25 HO RF relative.pkl',
                '0.5': '20 split 0.5 HO RF relative.pkl',
                '0.75': '20 split 0.75 HO RF relative.pkl'}

svm_abs_dict = {'0.05': '20 split 0.05 HO SVM absolute.pkl',
                '0.1': '20 split 0.1 HO SVM absolute.pkl',
                '0.15': '20 split 0.15 HO SVM absolute.pkl',
                '0.25': '20 split 0.25 HO SVM absolute.pkl',
                '0.5': '20 split 0.5 HO SVM absolute.pkl',
                '0.75': '20 split 0.75 HO SVM absolute.pkl'}

svm_rel_dict = {'0.05': '20 split 0.05 HO SVM relative.pkl',
                '0.1': '20 split 0.1 HO SVM relative.pkl',
                '0.15': '20 split 0.15 HO SVM relative.pkl',
                '0.25': '20 split 0.25 HO SVM relative.pkl',
                '0.5': '20 split 0.5 HO SVM relative.pkl',
                '0.75': '20 split 0.75 HO SVM relative.pkl'}

svm_abs_dict = {'0.05': '20 split 0.05 HO SVM absolute.pkl',
                '0.1': '20 split 0.1 HO SVM absolute.pkl',
                '0.15': '20 split 0.15 HO SVM absolute.pkl',
                '0.25': '20 split 0.25 HO SVM absolute.pkl',
                '0.5': '20 split 0.5 HO SVM absolute.pkl',
                '0.75': '20 split 0.75 HO SVM absolute.pkl'}

cnn_rel_dict = {'0.05': '20 split 0.05 HO CNN relative.pkl',
                '0.1': '20 split 0.1 HO CNN relative.pkl',
                '0.15': '20 split 0.15 HO CNN relative.pkl',
                '0.25': '20 split 0.25 HO CNN relative.pkl',
                '0.5': '20 split 0.5 HO CNN relative.pkl',
                '0.75': '20 split 0.75 HO CNN relative.pkl'}

cnn_abs_dict = {'0.05': '20 split 0.05 HO CNN absolute.pkl',
                '0.1': '20 split 0.1 HO CNN absolute.pkl',
                '0.15': '20 split 0.15 HO CNN absolute.pkl',
                '0.25': '20 split 0.25 HO CNN absolute.pkl',
                '0.5': '20 split 0.5 HO CNN absolute.pkl',
                '0.75': '20 split 0.75 HO CNN absolute.pkl'}

rf_var_regress = {'absolute': {0.04: {'raw': 0.57,
                                        'adj': 0.87},
                               0.25: {'raw': 0.51,
                                        'adj': 0.87},
                               0.50: {'raw': 0.36,
                                        'adj': 0.87},
                               0.75: {'raw': 0.17,
                                        'adj': 0.85}},
                  'relative': {0.04: {'raw': 0.44,
                                        'adj': 0.89},
                               0.25: {'raw': 0.31,
                                        'adj': 0.90},
                               0.50: {'raw': 0.20,
                                         'adj': 0.89},
                               0.75: {'raw': 0.09,
                                         'adj': 0.89}}}
        
sv_var_regress = {'absolute': {0.04: {'raw': 0.36,
                                        'adj': 0.87},
                               0.20: {'raw': 0.25,
                                        'adj': 0.86},
                               0.50: {'raw': 0.19,
                                        'adj': 0.87},
                               0.75: {'raw': 0.13,
                                        'adj': 0.87}},
                  'relative': {0.04: {'raw': 0.32,
                                        'adj': 0.82},
                               0.25: {'raw': 0.15,
                                        'adj': 0.81},
                               0.50: {'raw': 0.31,
                                         'adj': 0.87},
                               0.75: {'raw': 0.16,
                                         'adj': 0.86}}}

cn_var_regress = {'absolute': {0.04: {'raw': 0.32,
                                        'adj': 0.83},
                               0.25: {'raw': -0.18,
                                        'adj': 0.73},
                               0.50: {'raw': -0.09,
                                        'adj': 0.72},
                               0.75: {'raw': -0.12,
                                        'adj': 0.68}},
                  'relative': {0.04: {'raw': 0.46,
                                        'adj': 0.88},
                               0.25: {'raw': 0.38,
                                        'adj': 0.91},
                               0.50: {'raw': 0.24,
                                         'adj': 0.90},
                               0.75: {'raw': 0.21,
                                         'adj': 0.89}}}

gc_var_regress = {'absolute': {0.04: {'raw': 0.68,
                                        'adj': 0.89},
                               0.25: {'raw': 0.36,
                                        'adj': 0.90},
                               0.50: {'raw': 0.37,
                                        'adj': 0.89},
                               0.75: {'raw': 0.28,
                                        'adj': 0.85}},
                  'relative': {0.04: {'raw': 0.40,
                                        'adj': 0.91},
                               0.25: {'raw': 0.41,
                                        'adj': 0.80},
                               0.50: {'raw': 0.33,
                                         'adj': 0.88},
                               0.75: {'raw': 0.16,
                                         'adj': 0.86}}}

af_var_regress = {'absolute': {0.04: {'raw': 0.77,
                                        'adj': 0.94},
                               0.25: {'raw': 0.44,
                                        'adj': 0.90},
                               0.50: {'raw': 0.45,
                                        'adj': 0.90},
                               0.75: {'raw': 0.22,
                                        'adj': 0.86}},
                  'relative': {0.04: {'raw': 0.37,
                                        'adj': 0.91},
                               0.25: {'raw': 0.25,
                                        'adj': 0.85},
                               0.50: {'raw': 0.43,
                                         'adj': 0.86},
                               0.75: {'raw': 0.01,
                                         'adj': 0.83}}}

regression_split_plot_dicts = {'RF': rf_var_regress,
                            #    'SVM': sv_var_regress,
                               'CNN': cn_var_regress,
                            #    'GCN': gc_var_regress,
                               'AFP': af_var_regress}
