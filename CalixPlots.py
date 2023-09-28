"""
Top level function to call calixarene plots
"""

import CalixClustering as CC
import calix_visual_settings as CVS

frame = CC.create_umap_cluster_frame(CVS.calixarene_publication_cluster_dict,
                                     CVS.calixarene_publication_consistent_UMAP_dict,
                                     4,
                                     0.95)

CC.create_umap_scatter(frame,
                       CVS.calixarene_publication_umap_scatter_dict)

