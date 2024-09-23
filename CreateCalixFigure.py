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

default_setting_dict = {'color': 'orange',
                        'marker': 'o',
                        'opacity': 0.5,
                        'size': 25}
file_setting_dict = None

# calix_setting_dict = None

# peptide_setting_dict = None

peptide_setting_dict = {'H3K4': {'color': 'blue',
                             'marker': 'o',
                             'opacity': 0.5,
                             'size': 25},
                    'H3K4ac': {'color': 'red',
                                 'marker': 'o',
                                 'opacity': 0.5,
                                 'size': 25},
                    'H3K4me1': {'color': 'green',
                                    'marker': 'o',
                                    'opacity': 0.5,
                                    'size': 25},
                    'H3K4me2': {'color': 'purple',
                                    'marker': 'o',
                                    'opacity': 0.5,
                                    'size': 25},
                    'H3K4me3': {'color': 'yellow',
                                    'marker': 'o',
                                    'opacity': 0.5,
                                    'size': 25},
                    'H3K9me3': {'color': 'cyan',
                                    'marker': 'o',
                                    'opacity': 0.5,
                                    'size': 25},
                    'H3R2me2a': {'color': 'magenta',
                                    'marker': 'o',
                                    'opacity': 0.5,
                                    'size': 25},
                    'H3R2me2s': {'color': 'black',
                                    'marker': 'o',
                                    'opacity': 0.5,
                                    'size': 25}}

calix_setting_dict = {'A': {'color': 'blue',
                            'marker': 'o',
                            'opacity': 0.5,
                            'size': 25},
                        'B': {'color': 'red',
                                'marker': 'o',
                                'opacity': 0.5,
                                'size': 25},
                        'C': {'color': 'green',
                                'marker': 'o',
                                'opacity': 0.5,
                                'size': 25},
                        'D': {'color': 'purple',
                                'marker': 'o',
                                'opacity': 0.5,
                                'size': 25},
                        'E': {'color': 'yellow',
                                'marker': 'o',
                                'opacity': 0.5,
                                'size': 25},
                        'F': {'color': 'cyan',
                                'marker': 'o',
                                'opacity': 0.5,
                                'size': 25},
                        'P': {'color': 'magenta',
                                'marker': 'o',
                                'opacity': 0.5,
                                'size': 25}}
                            
VCV.multi_scatter_plot(pickle_file_folder='Results Dictionaries/',
                       list_of_pickle_files=['Rel CNN No F.pkl',],
                       file_setting_dict=file_setting_dict,
                       calix_setting_dict=calix_setting_dict,
                       peptide_setting_dict=peptide_setting_dict,
                       default_setting_dict=default_setting_dict,
                       plot_setting_dict=plot_setting_dict,
                       organize_by='host',
                       single_plot=False,
                       output_name='Rel CNN No F',
                       save_fig=False,
                       calculate_metrics=True,
                       group_hosts=True)