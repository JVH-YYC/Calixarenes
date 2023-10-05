"""
List of plotting settings for calixarene project
"""

calixarene_publication_consistent_UMAP_dict = {'metric': 'euclidean',
                                               'spread': 3}

calixarene_publication_cluster_dict = {'target_column_list': ['Label_0',
                                                           'Label_1'],
                                         'target_mode': 'exact',
                                         'data_column_list': ['3-10',],
                                         'data_mode': 'range',
                                         'csv_file_directory': './ClusterData/',
                                         'csv_file_name': 'Log data excluding non-binders.csv'}

calixarene_publication_umap_scatter_dict = {'plot_width': 3.2,
                                            'plot_height': 3.2,
                                        'marker_colors': {'Label_0': {'A': ['Oranges', 5, 3],
                                                                    'N': ['Greys', 5, 3],
                                                                    'B': ['Greens', 5, 3],
                                                                    'D': ['Reds', 5, 3],
                                                                    'C': ['Purples', 5, 3],
                                                                    'E': ['Blues', 5, 3]}},
                                      'marker_opacity': {'Label_0': {'A': 0.8,
                                                                    'N': 0.8,
                                                                    'B': 0.8,
                                                                    'D': 0.8,
                                                                    'C': 0.8,
                                                                    'E': 0.8}},
                                      'marker_type': {'Label_1': {'D': 'o',
                                                                  'A': 's',
                                                                  'N': 'x',
                                                                  'P': 'd',
                                                                  'B': 'v',
                                                                  'H': '^',
                                                                  'K': '<',
                                                                  'R': '>',
                                                                  'X': 'p'}}}


