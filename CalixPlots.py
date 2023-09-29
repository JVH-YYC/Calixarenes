"""
Top level function to call calixarene plots
"""

import CalixClustering as CC
import calix_visual_settings as CVS

frame = CC.create_umap_cluster_frame(CVS.calixarene_publication_cluster_dict,
                                     CVS.calixarene_publication_consistent_UMAP_dict,
                                     7,
                                     0.1)

CC.kmeans_umap_plot(frame,
                    n_clusters=3,
                    analysis_filename='avg per calix analysis.csv')


