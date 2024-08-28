"""
Visualization scripts for calixarene project.

All data is held in dictionaries with the following structure: {'host': {'peptide 1': (result),
'peptide 2': (result), ...}, 'host 2': {'peptide 1': (result), 'peptide 2': (result), ...}, ...}

Functions for creating AUROC plots, violin plots, predicted vs. actual scatters, as well as
numerical accuracy summaries (MSE, R^2, etc.) are included.

There is also the issue of 'accurate' vs. 'useful'. If a prediction has the correct trend of
adsorption predictions, but is off by a systematic error, this would be represented by a
slope close to 1, but an intercept far from 0. The is 'useful', but not 'accurate' (accurate will
have both slope and interecept within a certain range)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, auc, r2_score, mean_squared_error

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

def multi_scatter_plot(pickle_file_folder,
                       list_of_pickle_files,
                       translation_dict,
                       plot_setting_dict,
                       organize_by,
                       single_plots=True,
                       output_name=None,
                       save_fig=False):
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

    # Set up the figure
    plt.figure(figsize=(plot_setting_dict['fig_width'], plot_setting_dict['fig_height']))
    plt.xlabel(plot_setting_dict['x_label'], fontsize=plot_setting_dict['axis_font_size'])
    plt.ylabel(plot_setting_dict['y_label'], fontsize=plot_setting_dict['axis_font_size'])
    plt.xticks(fontsize=plot_setting_dict['tick_font_size'])
    plt.yticks(fontsize=plot_setting_dict['tick_font_size'])
    plt.title(plot_setting_dict['title'], fontsize=plot_setting_dict['title_font_size'])

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
            print(organized_dict['all'])
            predicted_values = [item['predicted'] for item in organized_dict['all']]
            actual_values = [item['actual'] for item in organized_dict['all']]

            # Extract the plot settings from translation_dict
            color = translation_dict['all'].get('color', 'blue')
            size = translation_dict['all'].get('size', 50)
            opacity = translation_dict['all'].get('opacity', 0.7)
            marker = translation_dict['all'].get('marker', 'o')

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
    # Currently working here **********
    return



        

