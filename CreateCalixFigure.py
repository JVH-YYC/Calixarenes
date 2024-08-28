"""
Top-level script for producing calixarene visualizations for publication

Translation dictionaries contained at this level - good settings can be saved in Visualization folder
"""

import Visualization.CalixViz as VCV

plot_setting_dict = {'fig_width': 10,
                     'fig_height': 10,
                     'x_label': 'Predicted ln(Kd)',
                     'y_label': 'Actual ln(Kd)',
                     'axis_font_size': 16,
                     'tick_font_size': 12,
                     'title_font_size': 20,
                     'title': 'Predicted vs. Actual ln(Kd) for Calixarene Hosts'}

translation_dict = {'all': {'color': 'orange',
                            'marker': 'o',
                            'opacity': 0.5,
                            'size': 25},}

VCV.multi_scatter_plot(pickle_file_folder='Results Dictionaries/',
                       list_of_pickle_files=['AttentiveFP_regression.pkl',],
                       translation_dict=translation_dict,
                       plot_setting_dict=plot_setting_dict,
                       organize_by='none',
                       single_plots=True,
                       output_name='first_AFP_test',
                       save_fig=False)