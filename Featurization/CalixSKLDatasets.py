"""
Benchmark ML scripts for calixarene evaluations
"""
import random
import numpy as np
import sklearn as skl
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor as RFR 
from Featurization import CalixSKLFeatures as CSF

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
                                                'ECFP': CSF.create_ecfp6_fingerprint(row['SMILES']),
                                                'Target_Val': tuple(row[target_columns])}
            elif target_columns_per_example == 'each':
                for targ_no, specific_column in enumerate(target_columns):
                    calixarene_dict[row['Host'] + str('_') + str(targ_no)] = {'SMILES': row['SMILES'],
                                                    'ECFP': CSF.create_ecfp6_fingerprint(row['SMILES']),
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

def cross_validation_split_calix_dataset(calixarene_dict,
                                         split_method,
                                         train_fraction,
                                         test_fraction,
                                         num_folds):
    """
    A function to split the calixarene benchmark dataset created by
    create_ecfp_dictionary into training, validation, and test sets,
    and cross_fold them with num_folds.
    
    The split can occur 2 different ways:
        'by_point': in this case all points are treated eqally and split by random shuffle
        'by_host': in this case, all points with a given host are in either the train, test, or validation set"""

    # Create a dictionary to hold the split data
    calixarene_cv_dict = {}
    for count in range(num_folds):
        calixarene_cv_dict['CV' + str(count)] = {}
        calixarene_cv_dict['CV' + str(count)]['train'] = {}
        calixarene_cv_dict['CV' + str(count)]['validation'] = {}
        calixarene_cv_dict['CV' + str(count)]['test'] = {}

    # Split the data by point
    if split_method == 'by_point':
        #Shuffle list of keys
        key_list = list(calixarene_dict.keys())
        random.shuffle(key_list)

        # Determine size for each fold
        fold_size = len(key_list) // num_folds

        # Create 'n_splits' folds
        folds = [key_list[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]

        # Allocate remaining keys to the last fold, if any
        if len(key_list) % num_folds != 0:
            folds[-1].extend(key_list[num_folds * fold_size:])

        for i in range(num_folds):
            target_dict_key = 'CV' + str(i)
            validation_keys = folds[i]
            test_keys = folds[(i + 1) % num_folds]  # Take the next fold in a circular manner
            train_keys = [key for j, fold in enumerate(folds) if j != i and j != (i + 1) % num_folds for key in fold]

            calixarene_cv_dict[target_dict_key]['train'] = {key: calixarene_dict[key] for key in train_keys}
            calixarene_cv_dict[target_dict_key]['validation'] = {key: calixarene_dict[key] for key in validation_keys}
            calixarene_cv_dict[target_dict_key]['test'] = {key: calixarene_dict[key] for key in test_keys}

    # Split the data by host
    elif split_method == 'by_host':
        # Create a set of unique hosts
        calix_host_set = {calix.split('_')[0] for calix in calixarene_dict}
        calix_host_list = list(calix_host_set)
        random.shuffle(calix_host_list)

        # Determine size for each fold of hosts
        fold_size = len(calix_host_list) // num_folds

        # Create 'num_folds' folds of hosts
        host_folds = [calix_host_list[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]
        if len(calix_host_list) % num_folds != 0:
            host_folds[-1].extend(calix_host_list[num_folds * fold_size:])

        # Create a dictionary to hold the cross-validation split data
        calixarene_cv_dict = {}
        for i in range(num_folds):
            target_dict_key = 'CV' + str(i)
            calixarene_cv_dict[target_dict_key] = {'train': {}, 'validation': {}, 'test': {}}

            validation_host_set = set(host_folds[i])
            test_host_set = set(host_folds[(i + 1) % num_folds])
            train_host_set = calix_host_set - validation_host_set - test_host_set

            for calix, value in calixarene_dict.items():
                host = calix.split('_')[0]
                if host in train_host_set:
                    calixarene_cv_dict[target_dict_key]['train'][calix] = value
                elif host in validation_host_set:
                    calixarene_cv_dict[target_dict_key]['validation'][calix] = value
                elif host in test_host_set:
                    calixarene_cv_dict[target_dict_key]['test'][calix] = value
    
    return calixarene_cv_dict
    


def organize_random_forest_input(split_calix_dataset,
                                 dataset_target_type,
                                 ordered_feature_list,
                                 peptide_one_hot_encoding):
    """
    A simple function that takes the training/validation/test dataset created above,
    and prepares the data for input into a random forest regressor. This involves
    selecting some of features out of a larger library and crafting them into objects
    of the appropriate shape to pass to scikit learn RF regressor
    
    Calculate and create the target values at the same time to keep things aligned"""

    # Create a dictionary to hold the data
    calixarene_rf_dict = {}
    calixarene_rf_dict['train'] = {}
    calixarene_rf_dict['validation'] = {}
    calixarene_rf_dict['test'] = {}

    type_list = ['train', 'validation', 'test']

    # Create a numpy array with the features
    # For each of the test, validation, train sets - concatenate the features that are
    # given in 'ordered_feature_list', and attach to the final peptide_one_hot_encoding
    # only if the dataset_target_type is 'each'

    if dataset_target_type == 'each':
        for dataset_split in type_list:
            full_sample_list = []
            full_sample_target_list = []
            for example in split_calix_dataset[dataset_split]:
                feature_list = []
                for ordered_feature in ordered_feature_list:
                    feature_list.append(split_calix_dataset[dataset_split][example][ordered_feature])
                
                feature_list.append(peptide_one_hot_encoding[split_calix_dataset[dataset_split][example]['Target']])
                
                full_sample_list.append(np.concatenate(feature_list, axis=0))
                full_sample_target_list.append(split_calix_dataset[dataset_split][example]['Target_Val'])

            calixarene_rf_dict[dataset_split]['features'] = np.array(full_sample_list)
            calixarene_rf_dict[dataset_split]['target'] = np.array(full_sample_target_list)

    
    elif dataset_target_type == 'all':
        for dataset_split in type_list:
            full_sample_list = []  # List to hold feature arrays
            full_sample_target_list = []  # List to hold target values
            for example in split_calix_dataset[dataset_split]:
                feature_list = []  # List to hold individual feature values
                for ordered_feature in ordered_feature_list:
                    feature_list.append(split_calix_dataset[dataset_split][example][ordered_feature])
                full_sample_list.append(np.concatenate(feature_list, axis=0))  # Convert list to numpy array and append
                full_sample_target_list.append(split_calix_dataset[dataset_split][example]['Target_Val'])
            calixarene_rf_dict[dataset_split]['features'] = np.array(full_sample_list)
            calixarene_rf_dict[dataset_split]['target'] = np.array(full_sample_target_list)
            
    return calixarene_rf_dict