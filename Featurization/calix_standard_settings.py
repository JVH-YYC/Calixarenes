"""
Python file to hold consistent settings for calixarene machine learning
"""
import numpy as np

# Peptide one_hot_encoding refers to modifications along peptide chain. Positions 2 and 4 have multiple, so the
# translation is 1-A; 2-R; 2-2mes; 2-2mea; 3-T; 4-K; 4-me1; 4-me2; 4-me3; 5-Q; 7-A;
# Values inferred by zero are 1-7; 2-A; 3-R; 4-ac; 5-S; 7-G 
peptide_one_hot_encoding = {'H3K4': np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1]),
                            'H3K4me1': np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1]),
                            'H3K4me2': np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1]),
                            'H3K4me3': np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1]),
                            'H3R2me2s': np.array([1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1]),
                            'H3R2me2a': np.array([1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]),
                            'H3K9me3': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
                            'H3K4ac': np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1])}
