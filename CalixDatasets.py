"""
Benchmark ML scripts for calixarene evaluations
"""

import sklearn as skl
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor as RFR 
from Featurization import CalixFeatures as CF

def create_ecfp_dictionary(calixarene_csv_folder,
                           calixarene_csv_file,
                           target_columns,
                           target_columns_per_example):
    """
    A function that reads a .csv containing calixarenes described
    as their smiles strings, and converts them into a dictionary.
    The .csv files have entries for every peptide measured, and these
    can be converted into a 7-position fingerprint for each calixarene,
    or they can be split out into individual entires for every example.
    
    Creates a dictionary of dictionaries. The first key is the calixarene
    name found in the 'Host' column. It points to a dictionary with keys 'SMILES',
    'ECFP', 'Target_Val', and optionally, 'Target'.
    
    If each peptide is being measured individually, then 'Target' is included.
    
    target_columns_per_example has 2 values: 'each' or 'all'.
    """

    # Read in the .csv file
    calixarene_df = pd.read_csv(calixarene_csv_folder + calixarene_csv_file)

    # Create a dictionary to hold the calixarene data
    calixarene_dict = {}

    # Iterate through the rows of the dataframe
    for index, row in calixarene_df.iterrows():
        #Check for duplicates
        if row['Host'] not in calixarene_dict:
            if target_columns_per_example == 'all':
                calixarene_dict[row['Host']] = {'SMILES': row['SMILES'],
                                                'ECFP': CF.create_ecfp6_fingerprint(row['SMILES']),
                                                'Target_Val': tuple(row[target_columns])}
            elif target_columns_per_example == 'each':
                for targ_no, specific_column in enumerate(target_columns):
                    calixarene_dict[row['Host'] + str('_') + str(targ_no)] = {'SMILES': row['SMILES'],
                                                    'ECFP': CF.create_ecfp6_fingerprint(row['SMILES']),
                                                    'Target_Val': row[specific_column],
                                                    'Target': specific_column}
    
    return calixarene_dict


