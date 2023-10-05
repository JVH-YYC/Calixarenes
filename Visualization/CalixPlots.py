"""
Top level function to call calixarene plots
"""

import Visualization.CalixClustering as CC
import Visualization.calix_visual_settings as CVS

frame = CC.create_umap_cluster_frame(CVS.calixarene_publication_cluster_dict,
                                     CVS.calixarene_publication_consistent_UMAP_dict,
                                     5,
                                     0.05)

CC.kmeans_umap_plot(frame,
                    n_clusters=3,
                    analysis_filename='raw log data analysis.csv')


