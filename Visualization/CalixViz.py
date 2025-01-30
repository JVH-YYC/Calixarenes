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

import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import statistics

from sklearn.metrics import roc_curve, auc, r2_score, mean_squared_error
from scipy.stats import pearsonr

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
                    curr_predict = [x[0] for x in curr_results]
                    curr_actual = [x[1] for x in curr_results]
                    mse, r2, adjusted_r2, shift_amount = calculate_metrics(curr_predict, curr_actual)
                    all_r2.append(r2)
                    all_adj_r2.append(adjusted_r2)
        
        abs_r2_success = sum([1 for x in all_r2 if x > 0.6]) / len(all_r2)
        adj_r2_success = sum([1 for x, y in zip(all_r2, all_adj_r2) if x <= 0.6 and y > 0.6]) / len(all_r2)
        results_dict[holdout_amt]['r2_median'] = statistics.median(all_r2)
        results_dict[holdout_amt]['r2_success'] = abs_r2_success
        results_dict[holdout_amt]['adj_r2_median'] = statistics.median(all_adj_r2)
        results_dict[holdout_amt]['adj_r2_success'] = adj_r2_success
        results_dict[holdout_amt]['summed_success'] = abs_r2_success + adj_r2_success
    return results_dict

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



        

