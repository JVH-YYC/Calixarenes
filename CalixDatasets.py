"""
Benchmark ML scripts for calixarene evaluations
"""
import random
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

def split_calix_dataset(calixarene_dict,
                        split_method,
                        train_fraction,
                        test_fraction):
    """
    A function to split the calixarene benchmark dataset created by
    create_ecfp_dictionary into training, validation, and test sets.
    
    The split can occur 2 different ways:
        'by_point': in this case all points are treated eqally and split by random shuffle
        'by_host': in this case, all points with a given host are in either the train, test, or validation set"""

    # Create a dictionary to hold the split data
    calixarene_split_dict = {}
    calixarene_split_dict['train'] = {}
    calixarene_split_dict['validation'] = {}
    calixarene_split_dict['test'] = {}


    # Split the data by point
    if split_method == 'by_point':
        #Shuffle list of keys
        key_list = list(calixarene_dict.keys())
        random.shuffle(key_list)

        for specific_key in key_list:
            # Check if the training set is full
            if len(calixarene_split_dict['train']) < train_fraction * len(calixarene_dict):
                calixarene_split_dict['train'][specific_key] = calixarene_dict[specific_key]
            # Check if the validation set is full
            elif (len(calixarene_split_dict['train']) + len(calixarene_split_dict['test'])) < (train_fraction + test_fraction) * len(calixarene_dict):
                calixarene_split_dict['test'][specific_key] = calixarene_dict[specific_key]
            # If neither of the above are true, then the validation set is filled
            else:
                calixarene_split_dict['validation'][specific_key] = calixarene_dict[specific_key]

    # Split the data by host
    elif split_method == 'by_host':
        # Create a set of unique hosts
        calix_host_list = []
        for calix in calixarene_dict:
            if calix.split('_')[0] not in calix_host_list:
                calix_host_list.append(calix.split('_')[0])
        random.shuffle(calix_host_list)

        # Split hosts into training, validation, and test sets
        train_host_set = set(calix_host_list[:int(train_fraction * len(calix_host_list))])
        test_host_set = set(calix_host_list[int(train_fraction * len(calix_host_list)):int((train_fraction + test_fraction) * len(calix_host_list))])
        validation_host_set = set(calix_host_list[int((train_fraction + test_fraction) * len(calix_host_list)):])

        # Loop back through dictionary, and now use the host set to split the data
        for calix in calixarene_dict:
            if calix.split('_')[0] in train_host_set:
                calixarene_split_dict['train'][calix] = calixarene_dict[calix]
            elif calix.split('_')[0] in validation_host_set:
                calixarene_split_dict['validation'][calix] = calixarene_dict[calix]
            elif calix.split('_')[0] in test_host_set:
                calixarene_split_dict['test'][calix] = calixarene_dict[calix]
    
    return calixarene_split_dict
