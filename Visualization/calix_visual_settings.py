"""
List of plotting settings for calixarene project
"""
import numpy as np

calixarene_publication_consistent_UMAP_dict = {'metric': 'euclidean',
                                               'spread': 3}

calixarene_publication_cluster_dict = {'target_column_list': ['Label_0',
                                                           'Label_1'],
                                         'target_mode': 'exact',
                                         'data_column_list': ['3-10',],
                                         'data_mode': 'range',
                                         'csv_file_directory': './ClusterData/',
                                         'csv_file_name': 'Log data excluding non-binders.csv'}

calixarene_publication_umap_scatter_dict = {'plot_width': 5,
                                            'plot_height': 5,
                                        'calix_color': {'A': ['Oranges', 5, 3],
                                                        'P': ['Greys', 5, 3],
                                                        'B': ['Greens', 5, 3],
                                                        'D': ['Reds', 5, 3],
                                                        'C': ['Purples', 5, 3],
                                                        'E': ['Blues', 5, 3]},
                                      'marker_opacity': 0.8,
                                      'marker_type': 'o'}

calix_pub_heatmap_settings = {'plot_width': 6,
                        'plot_height': 9.25,
                        'log_val': True,
                        'font_type': 'DejaVu Sans',
                        'y_font_size': 7,
                        'x_font_size': 8,
                        'tick_font_size': 10,
                        'tick_override': True,
                        # Raw data is in µM - conversion to M means ticks are off by e6
                        'tick_positions': [np.log10(1e2),
                                           np.log10(1e1), 
                                           np.log10(1),
                                           np.log10(1e-1),
                                           np.log10(1e-2),
                                           np.log10(1e-3)],
                        'tick_labels': [r'$10^{-4}$',
                                        r'$10^{-5}$',
                                        r'$10^{-6}$',
                                        r'$10^{-7}$',
                                        r'$10^{-8}$',
                                        r'$10^{-9}  $']}



