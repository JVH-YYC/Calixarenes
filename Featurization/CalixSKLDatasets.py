"""
Benchmark ML scripts for calixarene evaluations
"""
import os
import platform
import random
import itertools
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

    Parameters
    ----------
    calixarene_csv_folder : string
        Self-evident - name of folder containing calixarene CSV file
    calixarene_csv_file : string
        Self-evident - name of file containing calixarene CSV file
    target_columns : list of strings
        If only certain peptides will be evaluated, this list will match the target columns in the adsorption .csv
        file - aka peptide names (H3K4, etc.)
    target_columns_per_example: string ('each' or 'all')
        Determines whether 'all' peptide targets for a host will be partitioned into the same training/test set, or whether
        'each' data point can be moved individually. From the point of view of someone trying to make new calixarenes, 'all'
        is the correct choice. An entire calixarene needs to be held out for testing, not just a single point.
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

def create_structured_ecfp_dictionary(calixarene_csv_folder,
                                      calixarene_csv_file,
                                      split_calixarene_dict,
                                      holdout_size):
    """
    A function used to test different sizes of data holdout - will there be a difference between
    absolute training and relative training?

    For the input calixarene lists, they are pre-segregated into the types of calixarenes that have been observed
    to be 'predictable' (mono and unsubstituted) and 'unpredictable' (multiple substituents)

    We only care about predictions for 'predictable' systems, so we exclude the 'unpredictable' systems from the test set.

    Parameters
    ----------
    calixarene_csv_folder : string
        Self-evident - name of folder containing calixarene CSV file
    calixarene_csv_file : string
        Self-evident - name of file containing calixarene CSV file
    split_calixarene_dict: dictionary
        Dict with keys of 'predictable' and 'unpredictable': the lab-generated dataset is quite unevenly split between
        mono- or unsubstituted calixarenes (e.g. P, A, E-type) vs multi-substituted (B, C, D), so in some analyses these
        are considered separately.
    holdout_size : float
        Determines size of training/test split as one additional evaluation    
    """

    # Read in the .csv file
    calixarene_df = pd.read_csv(calixarene_csv_folder + calixarene_csv_file)

    # Create a dictionary to hold the calixarene data
    calixarene_dict = {}
    calixarene_dict['train'] = {}
    calixarene_dict['test'] = {}

    # Determine the holdout calixarenes
    holdout_calixarenes_pred = random.sample(list(split_calixarene_dict['predictable']), holdout_size)
    holdout_calixarenes_unpred = random.sample(list(split_calixarene_dict['unpredictable']), holdout_size)

    # Always looking at all columns
    target_columns = ['H3K4', 'H3K4ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9me3', 'H3R2me2a', 'H3R2me2s']
    # Iterate through the rows of the dataframe
    for index, row in calixarene_df.iterrows():
        #Check for duplicates and whether the host is the holdout
        if row['Host'] not in holdout_calixarenes_pred and row['Host'] not in holdout_calixarenes_unpred:
            if row['Host'] not in calixarene_dict['train']:
                for targ_no, specific_column in enumerate(target_columns):
                    calixarene_dict['train'][row['Host'] + str('_') + str(targ_no)] = {'SMILES': row['SMILES'],
                                                    'ECFP': CSF.create_ecfp6_fingerprint(row['SMILES']),
                                                    'Target_Val': row[specific_column],
                                                    'Target': specific_column}
        else:
            if row['Host'] not in calixarene_dict['test'] and row['Host'] in holdout_calixarenes_pred:
                for targ_no, specific_column in enumerate(target_columns):
                    calixarene_dict['test'][row['Host'] + str('_') + str(targ_no)] = {'SMILES': row['SMILES'],
                                                    'ECFP': CSF.create_ecfp6_fingerprint(row['SMILES']),
                                                    'Target_Val': row[specific_column],
                                                    'Target': specific_column}
    
    return calixarene_dict

def create_structured_relative_ecfp_dictionary(calixarene_csv_folder,
                                               calixarene_csv_file,
                                               split_calixarene_dict,
                                               holdout_size):
    """
    A complementary function to that directly above, just for relative training/prediction rather than absolute

    Only method being used for fingerprints is 'concat', based on previous testing

    To this dict, add 'test calix' as a key, with the list of selected calixarenes for testing. Unlike in absolute training,
    it's not immediately obvious which calix is being tested.

    Parameters are identical to that above.
    """

    # Read in the .csv file
    calixarene_df = pd.read_csv(calixarene_csv_folder + calixarene_csv_file)

    calixarene_comparison_dict = {}
    calixarene_comparison_dict['train'] = {}
    calixarene_comparison_dict['test'] = {}

    # Always looking at all columns
    target_columns = ['H3K4', 'H3K4ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9me3', 'H3R2me2a', 'H3R2me2s']

    # Determine the holdout calixarenes - int() always rounds down
    holdout_pred_amount = int(len(split_calixarene_dict['predictable']) * holdout_size)
    holdout_unpred_amount = int(len(split_calixarene_dict['unpredictable']) * holdout_size)

    holdout_calixarenes_pred = random.sample(split_calixarene_dict['predictable'], holdout_pred_amount)
    holdout_calixarenes_unpred = random.sample(split_calixarene_dict['unpredictable'], holdout_unpred_amount)
    all_holdout_calix = holdout_calixarenes_pred + holdout_calixarenes_unpred
    calixarene_comparison_dict['holdout'] = all_holdout_calix

    # Iterate over all combinations of two different hosts
    for (idx1, row1), (idx2, row2) in itertools.permutations(calixarene_df.iterrows(), 2):
        host_pair = (row1['Host'], row2['Host'])
        
        if row1['Host'] not in all_holdout_calix and row2['Host'] not in all_holdout_calix:
            for target in target_columns:
                key = host_pair + (target,)
                calixarene_comparison_dict['train'][key] = {'SMILES': (row1['SMILES'], row2['SMILES']),
                                                         'ECFP': CSF.create_double_ecpf6_fingerprint((row1['SMILES'], row2['SMILES']),
                                                                                                     method='concat'),
                                                         'Target_Val': row1[target] - row2[target],
                                                         'Target': target}
        else:
            if (row1['Host'] in holdout_calixarenes_pred) ^ (row2['Host'] in holdout_calixarenes_pred):
                if row1['Host'] not in holdout_calixarenes_unpred and row2['Host'] not in holdout_calixarenes_unpred:
                    for target in target_columns:
                        key = host_pair + (target,)
                        calixarene_comparison_dict['test'][key] = {'SMILES': (row1['SMILES'], row2['SMILES']),
                                                                'ECFP': CSF.create_double_ecpf6_fingerprint((row1['SMILES'], row2['SMILES']),
                                                                                                            method='concat'),
                                                                'Target_Val': row1[target] - row2[target],
                                                                'Target': target}
                        if row1['Host'] in holdout_calixarenes_pred:
                            calixarene_comparison_dict['test'][key]['test_pos'] = 'row1'
                            calixarene_comparison_dict['test'][key]['known_val'] = row2[target]
                        elif row2['Host'] in holdout_calixarenes_pred:
                            calixarene_comparison_dict['test'][key]['test_pos'] = 'row2'
                            calixarene_comparison_dict['test'][key]['known_val'] = row1[target]

    return calixarene_comparison_dict

def create_structured_absolute_ecfp_dictionary(calixarene_csv_folder,
                                               calixarene_csv_file,
                                               split_calixarene_dict,
                                               holdout_size):
    """
    A complementary function to that above, but for absolute training/prediction rather than relative

    Add held-out calixarene names to the dictionary for later use

    Over time, this evolved into a duplicate of the function 2 above.
    """

    # Read in the .csv file
    calixarene_df = pd.read_csv(calixarene_csv_folder + calixarene_csv_file)

    calixarene_dict = {}
    calixarene_dict['train'] = {}
    calixarene_dict['test'] = {}

    # Always looking at all columns
    target_columns = ['H3K4', 'H3K4ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9me3', 'H3R2me2a', 'H3R2me2s']

    # Determine the holdout calixarenes - int() always rounds down
    holdout_pred_amount = int(len(split_calixarene_dict['predictable']) * holdout_size)
    holdout_unpred_amount = int(len(split_calixarene_dict['unpredictable']) * holdout_size)

    holdout_calixarenes_pred = random.sample(split_calixarene_dict['predictable'], holdout_pred_amount)
    holdout_calixarenes_unpred = random.sample(split_calixarene_dict['unpredictable'], holdout_unpred_amount)
    all_holdout_calix = holdout_calixarenes_pred + holdout_calixarenes_unpred
    # Iterate through the rows of the dataframe
    for index, row in calixarene_df.iterrows():
        #Check for duplicates and whether the host is the holdout
        if row['Host'] not in all_holdout_calix:
            if row['Host'] not in calixarene_dict['train']:
                for targ_no, specific_column in enumerate(target_columns):
                    calixarene_dict['train'][row['Host'] + str('_') + str(targ_no)] = {'SMILES': row['SMILES'],
                                                    'ECFP': CSF.create_ecfp6_fingerprint(row['SMILES']),
                                                    'Target_Val': row[specific_column],
                                                    'Target': specific_column}
        else:
            if row['Host'] not in calixarene_dict['test']:
                for targ_no, specific_column in enumerate(target_columns):
                    calixarene_dict['test'][row['Host'] + str('_') + str(targ_no)] = {'SMILES': row['SMILES'],
                                                    'ECFP': CSF.create_ecfp6_fingerprint(row['SMILES']),
                                                    'Target_Val': row[specific_column],
                                                    'Target': specific_column}
    
    return calixarene_dict
    
def create_loo_ecfp_dictionary(calixarene_csv_folder,
                               calixarene_csv_file,
                               holdout_calixarene):
    """
    Essentially identical to the function above, but serves a singular purpose: making the final LOO dictionary
    for final benchmarking. Only done in 'host' mode, and with each peptide predicted individually with one-hot encoding.

    This function will *skip* the dataset splitting step, so test and train are created here.
    """

    # Read in the .csv file for calixarenes and one-hot encodings
    calixarene_df = pd.read_csv(calixarene_csv_folder + calixarene_csv_file)

    # Create a dictionary to hold the calixarene data
    calixarene_dict = {}
    calixarene_dict['train'] = {}
    calixarene_dict['test'] = {}

    # Always looking at all columns
    target_columns = ['H3K4', 'H3K4ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9me3', 'H3R2me2a', 'H3R2me2s']
    # Iterate through the rows of the dataframe
    for index, row in calixarene_df.iterrows():
        #Check for duplicates and whether the host is the holdout
        if row['Host'] != holdout_calixarene:
            if row['Host'] not in calixarene_dict['train']:
                for targ_no, specific_column in enumerate(target_columns):
                    calixarene_dict['train'][row['Host'] + str('_') + str(targ_no)] = {'SMILES': row['SMILES'],
                                                    'ECFP': CSF.create_ecfp6_fingerprint(row['SMILES']),
                                                    'Target_Val': row[specific_column],
                                                    'Target': specific_column}
        else:
            if row['Host'] not in calixarene_dict['test']:
                for targ_no, specific_column in enumerate(target_columns):
                    calixarene_dict['test'][row['Host'] + str('_') + str(targ_no)] = {'SMILES': row['SMILES'],
                                                    'ECFP': CSF.create_ecfp6_fingerprint(row['SMILES']),
                                                    'Target_Val': row[specific_column],
                                                    'Target': specific_column}
    
    return calixarene_dict

def create_relative_ecfp_dictionary(calixarene_csv_folder,
                                    calixarene_csv_file,
                                    target_columns,
                                    target_columns_per_example):
    """
    A function that reads the data .csv, and then creates a dictionary
    comparing each unique pair of calixarenes, and records the difference
    between the two dataframe values.
    """
    # Read in the .csv file
    calixarene_df = pd.read_csv(calixarene_csv_folder + calixarene_csv_file)

    calixarene_comparison_dict = {}

    # Iterate over all combinations of two different hosts
    for (idx1, row1), (idx2, row2) in itertools.combinations(calixarene_df.iterrows(), 2):
        host_pair = (row1['Host'], row2['Host'])
        
        if target_columns_per_example == 'each':
            for target in target_columns:
                key = host_pair + (target,)
                calixarene_comparison_dict[key] = {'SMILES': (row1['SMILES'], row2['SMILES']),
                                                         'ECFP': CSF.create_double_ecpf6_fingerprint((row1['SMILES'], row2['SMILES'])),
                                                         'Target_Val': row1[target] - row2[target],
                                                         'Target': target}
        elif target_columns_per_example == 'all':
            differences = tuple(row1[target] - row2[target] for target in target_columns)
            calixarene_comparison_dict[key] = {'SMILES': (row1['SMILES'], row2['SMILES']),
                                                     'ECFP': CSF.create_double_ecpf6_fingerprint((row1['SMILES'], row2['SMILES'])),
                                                     'Target_Val': differences}

    return calixarene_comparison_dict

def create_loo_relative_ecfp_dictionary(calixarene_csv_folder,
                                        calixarene_csv_file,
                                        holdout_calixarene,
                                        method):
    """
    A nearly identical function to that above, but for the final LOO benchmarking,
    so always in 'host' and 'each' mode.

    As with the absolute LOO dictionary, there will be no splitting happening later on - so 'test' and 'train' are set up immediately.
    """

    # Read in the .csv file
    calixarene_df = pd.read_csv(calixarene_csv_folder + calixarene_csv_file)

    calixarene_comparison_dict = {}
    calixarene_comparison_dict['train'] = {}
    calixarene_comparison_dict['test'] = {}

    # Always looking at all columns
    target_columns = ['H3K4', 'H3K4ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9me3', 'H3R2me2a', 'H3R2me2s']

    # Iterate over all combinations of two different hosts
    for (idx1, row1), (idx2, row2) in itertools.permutations(calixarene_df.iterrows(), 2):
        host_pair = (row1['Host'], row2['Host'])
        
        if row1['Host'] != holdout_calixarene and row2['Host'] != holdout_calixarene:
            for target in target_columns:
                key = host_pair + (target,)
                calixarene_comparison_dict['train'][key] = {'SMILES': (row1['SMILES'], row2['SMILES']),
                                                         'ECFP': CSF.create_double_ecpf6_fingerprint((row1['SMILES'], row2['SMILES']),
                                                                                                     method),
                                                         'Target_Val': row1[target] - row2[target],
                                                         'Target': target}
        else:
            for target in target_columns:
                key = host_pair + (target,)
                calixarene_comparison_dict['test'][key] = {'SMILES': (row1['SMILES'], row2['SMILES']),
                                                         'ECFP': CSF.create_double_ecpf6_fingerprint((row1['SMILES'], row2['SMILES']),
                                                                                                     method),
                                                         'Target_Val': row1[target] - row2[target],
                                                         'Target': target}
                if row1['Host'] == holdout_calixarene:
                    calixarene_comparison_dict['test'][key]['test_pos'] = 'row1'
                    calixarene_comparison_dict['test'][key]['known_val'] = row2[target]
                elif row2['Host'] == holdout_calixarene:
                    calixarene_comparison_dict['test'][key]['test_pos'] = 'row2'
                    calixarene_comparison_dict['test'][key]['known_val'] = row1[target]

    return calixarene_comparison_dict

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
        'by_host': in this case, all points with a given host are in either the train, test, or validation set
    """

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
        calix_host_set = set()
        
        # Create a set of unique hosts. Depends on training type (absolute or relative)
        first_key = next(iter(calixarene_dict))
        if type(first_key) == str:
            calix_host_set = {calix.split('_')[0] for calix in calixarene_dict}
        elif type(first_key) == tuple:
            calix_host_set = {host.split('_')[0] for calix in calixarene_dict for host in calix}
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
                if type(calix) == str:
                    host = calix.split('_')[0]
                    if host in train_host_set:
                        calixarene_cv_dict[target_dict_key]['train'][calix] = value
                    elif host in validation_host_set:
                        calixarene_cv_dict[target_dict_key]['validation'][calix] = value
                        print('Added to validation set:', calix)
                    elif host in test_host_set:
                        calixarene_cv_dict[target_dict_key]['test'][calix] = value
                        print('Added to test set:', calix)
                elif type(calix) == tuple:
                    host1 = calix[0].split('_')[0]
                    host2 = calix[1].split('_')[0]
                    if host1 in train_host_set and host2 in train_host_set:
                        calixarene_cv_dict[target_dict_key]['train'][calix] = value
                    elif host1 in validation_host_set or host2 in validation_host_set:
                        calixarene_cv_dict[target_dict_key]['validation'][calix] = value
                        print('Added to validation set:', calix)
                    elif host1 in test_host_set or host2 in test_host_set:
                        calixarene_cv_dict[target_dict_key]['test'][calix] = value
                        print('Added to test set:', calix)

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
    
    Calculate and create the target values at the same time to keep things aligned
    
    Parameters
    ----------
    split_calix_dataset: dictionary
        For any of the several above functions, a dictionary of training/test examples is created - that becomes input here
    dataset_target_type: string ('all' or 'each')
        Identical to that above: dataset dictionaries are created in 'all' or 'each' mode, must be consistent here.
    ordered_feature_list: list of ECFP6 fingerprint created above 
        ECFP6 fingerprint that is concatenated w/ peptide information here to craft a random forest input
    peptide_one_hot_encoding: dictionary
        A dictionary that contains peptide names as keys, and points to one-hot-encoded vectories of their properties

    """



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

def organize_loo_model_input(loo_calix_dataset,
                             one_hot_encoding_folder,
                             peptide_one_hot_encoding,
                             relative_training):
    """
    A function derived from the one directly above, but with some features removed as it is only for final LOO testing.

    Therefore, there is no 'validation' split, and the data is always split by 'host', and each point is treated individually.

    Also, the only ordered feature for the final investigation is 'EFCP', so the ordered_feature_list is removed.

    Parameters
    ----------
    loo_calix_dataset: dictionary
        A training/test example dictionary created by one of LOO creation functions above
    one_hot_encoding_folder : string
        A string that points to the folder holding one hot encoding CSV
    peptide_one_hot_encoding : string
        Actual file name for the one-hot encoding CSV
    relative_training : Boolean
        Indicates whether training type is relative (True) or absolute (False)
    """
    # Open one-hot encodings as dataframe
    one_hot_df = pd.read_csv(one_hot_encoding_folder + peptide_one_hot_encoding, index_col='Peptide')

    calixarene_model_dict = {}
    calixarene_model_dict['train'] = {}
    calixarene_model_dict['test'] = {}

    type_list = ['train', 'test']
    peptide_name_order = []

    for dataset_split in type_list:
        full_sample_list = []
        full_sample_target_list = []
        if relative_training and dataset_split == 'test':
            test_calix_position = []
            known_calix_value = []
            full_peptide_list = []

        for idx, example in enumerate(loo_calix_dataset[dataset_split]):
            if idx < 8 and dataset_split == 'test':
                #8 peptides in final LOO - record the first 8 in order
                peptide_name_order.append(loo_calix_dataset[dataset_split][example]['Target'])
            
            feature_list = []
            feature_list.append(loo_calix_dataset[dataset_split][example]['ECFP'])
            feature_list.append(list(one_hot_df.loc[loo_calix_dataset[dataset_split][example]['Target']]))
            
            full_sample_list.append(np.concatenate(feature_list, axis=0))
            full_sample_target_list.append(loo_calix_dataset[dataset_split][example]['Target_Val'])
            if relative_training and dataset_split == 'test':
                test_calix_position.append(loo_calix_dataset[dataset_split][example]['test_pos'])
                known_calix_value.append(loo_calix_dataset[dataset_split][example]['known_val'])
                full_peptide_list.append(loo_calix_dataset[dataset_split][example]['Target'])

        calixarene_model_dict[dataset_split]['features'] = np.array(full_sample_list)
        calixarene_model_dict[dataset_split]['target'] = np.array(full_sample_target_list)
        if relative_training and dataset_split == 'test':
            calixarene_model_dict[dataset_split]['test_pos'] = test_calix_position
            calixarene_model_dict[dataset_split]['known_val'] = known_calix_value
            calixarene_model_dict[dataset_split]['peptide_order'] = full_peptide_list

    return calixarene_model_dict, peptide_name_order

def organize_structured_absolute_model_input(structured_calix_dataset,
                                    one_hot_encoding_folder,
                                    peptide_one_hot_encoding):
    """
    A complementary function to that above - with the main difference being that this example
    will have multiple calixarenes held out. Doesn't matter for training set - but test set must
    be organized in such a way that absolute R2 and adjusted R2 can be calculated.

    While not computationally efficient, directly porting the test set from the structured_calix_dataset
    would keep a useful structure
    """

    # Open one-hot encodings as dataframe
    one_hot_df = pd.read_csv(one_hot_encoding_folder + peptide_one_hot_encoding, index_col='Peptide')

    calixarene_model_dict = {}
    calixarene_model_dict['train'] = {}
    calixarene_model_dict['test'] = {}

    #Only process into training set
    full_sample_list = []
    full_sample_target_list = []

    for idx, example in enumerate(structured_calix_dataset['train']):
        feature_list = []
        feature_list.append(structured_calix_dataset['train'][example]['ECFP'])
        feature_list.append(list(one_hot_df.loc[structured_calix_dataset['train'][example]['Target']]))
        
        full_sample_list.append(np.concatenate(feature_list, axis=0))
        full_sample_target_list.append(structured_calix_dataset['train'][example]['Target_Val'])

        calixarene_model_dict['train']['features'] = np.array(full_sample_list)
        calixarene_model_dict['train']['target'] = np.array(full_sample_target_list)
    
    for idx, example in enumerate(structured_calix_dataset['test']):
        calixarene_model_dict['test'][example] = {}
        calixarene_model_dict['test'][example]['ECFP'] = structured_calix_dataset['test'][example]['ECFP']
        calixarene_model_dict['test'][example]['Target_Val'] = structured_calix_dataset['test'][example]['Target_Val']
        calixarene_model_dict['test'][example]['Peptide_OH'] = np.array(list(one_hot_df.loc[structured_calix_dataset['test'][example]['Target']]))
        
    return calixarene_model_dict

def organize_structured_relative_model_input(structured_calix_dataset,
                                              one_hot_encoding_folder,
                                              peptide_one_hot_encoding):
    """
    A complementary function to that above, but for relative training. When relative training,
    the dictionary contains a list of test calixarenes to aid in assembling the test results
    """

    # Open one-hot encodings as dataframe
    one_hot_df = pd.read_csv(one_hot_encoding_folder + peptide_one_hot_encoding, index_col='Peptide')

    calixarene_model_dict = {}
    calixarene_model_dict['train'] = {}
    calixarene_model_dict['test'] = {}
    calixarene_model_dict['holdout'] = structured_calix_dataset['holdout']
    
    #Only process into training set
    full_sample_list = []
    full_sample_target_list = []

    for idx, example in enumerate(structured_calix_dataset['train']):
        feature_list = []
        feature_list.append(structured_calix_dataset['train'][example]['ECFP'])
        feature_list.append(list(one_hot_df.loc[structured_calix_dataset['train'][example]['Target']]))
        
        full_sample_list.append(np.concatenate(feature_list, axis=0))
        full_sample_target_list.append(structured_calix_dataset['train'][example]['Target_Val'])

        calixarene_model_dict['train']['features'] = np.array(full_sample_list)
        calixarene_model_dict['train']['target'] = np.array(full_sample_target_list)
    
    for idx, example in enumerate(structured_calix_dataset['test']):
        calixarene_model_dict['test'][example] = structured_calix_dataset['test'][example]
        calixarene_model_dict['test'][example]['Peptide_OH'] = np.array(list(one_hot_df.loc[structured_calix_dataset['test'][example]['Target']]))
        
    return calixarene_model_dict
