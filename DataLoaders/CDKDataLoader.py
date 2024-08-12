#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 09:26:51 2020

@author: jvh

Data generation from single large .pq file of molecules and .csv file of experimental binding constants

"""
import pandas as pd
import random
from pathlib import Path
import numpy as np
import torch

def coordinate_load(pq_file_name,
                    pq_file_directory):
    """
    Takes names of parquet file
    Loads the data file into pandas frame
    Creates and returns a dictionary of file name associated with pandas frame    

    Parameters
    ----------
    pq_file_name : string
        File names to load into dataframe
    pq_file_directory : string
        The name of the directory that contains the parquet files
    
    Returns
    -------
    A dataframe with 3D inhibitor data

    """
    pq_path = Path('.', pq_file_directory)
    molecule_frame = pd.read_parquet(pq_path / pq_file_name)
    
    return molecule_frame

def calixarene_list(molecule_frame,
                    test_list):
    """
    Takes a populated molecular DataFrame and looks at all leading prefixes
    Creates a list of all observed prefixes, but will not include any prefix
    included in the 'test' list
    Returns a list of prefixes for further operations

    Parameters
    ----------
    molecule_frame : pandas DataFrame
        a molecular frame with ASO/POS/NEG/POL values
    test_list : list of strings
        A list of calixarenes that are held out for final testing

    Returns
    -------
    List of prefix strings to be used in training/test sets (not validation set)

    """

    column_list = molecule_frame.columns
    prefix_list = []
    
    for entry in column_list:
        counter = None
        underscore = False
        
        #Count backwards to find final underscore
        for character in range(len(entry), 0, -1):
            if entry[(character - 1)] == '_':
                counter = character
                underscore = True
                break

        if underscore is False:
            print('No underscore detected in entry:', entry)
            
        if underscore == True:
            #All data has extra label '_U' and '_D' for flipped conformers, so remove these from label
            current_prefix = entry[0:(counter - 3)]
            if current_prefix not in test_list and current_prefix not in prefix_list:
                prefix_list.append(current_prefix)
    return prefix_list

def labelled_example_generator(calix_1,
                               calix_2,
                               data_frame):
    """
    Takes two input prefixes, and uses a random number generator (with integers up to 'analogs')
    to create a 'difference' frame for the two calixarene molecules.
    This merges with the differential adsorption, which is generated with the labels when called.

    Parameters
    ----------
    calix_1 : string
        Prefix of the first calixarene
    calix_2 : string
        Prefix of the second calixarene
    analogs : integer
        The number of parallel repeat frames that were created
    data_frame : pandas DataFrame
        The single dataframe that contains ASO/NEG/POS/POL values for all calixarenes

    Returns
    -------
    A 'difference' frame for the given calixarene pair

    """
    
    first_random = random.getrandbits(1)
    
    if first_random == 0:
        this_pass_1 = 'U'
    else:
        this_pass_1 = 'D'
    
    #Create calixarene 1 dataframe

    cal_1_aso = calix_1 + '_' + this_pass_1 + '_ASO'
    cal_1_pos = calix_1 + '_' + this_pass_1 + '_POS'
    cal_1_neg = calix_1 + '_' + this_pass_1 + '_NEG'
    cal_1_pol = calix_1 + '_' + this_pass_1 + '_POL'

    calix_1_frame = pd.DataFrame({'x': data_frame['x'],
                                  'y': data_frame['y'],
                                  'z': data_frame['z'],
                                  'ASO': data_frame[cal_1_aso],
                                  'POS': data_frame[cal_1_pos],
                                  'NEG': data_frame[cal_1_neg],
                                  'POL': data_frame[cal_1_pol]})
    
    sec_random = random.getrandbits(1)
    
    if sec_random == 0:
        this_pass_2 = 'U'
    else:
        this_pass_2 = 'D'
        
    #Get calixarene 2 names
    cal_2_aso = calix_2 + '_' + this_pass_2 + '_ASO'
    cal_2_pos = calix_2 + '_' + this_pass_2 + '_POS'
    cal_2_neg = calix_2 + '_' + this_pass_2 + '_NEG'
    cal_2_pol = calix_2 + '_' + this_pass_2 + '_POL'

    #Create difference frame
    calix_1_frame['ASO'] -= data_frame[cal_2_aso]
    calix_1_frame['POS'] -= data_frame[cal_2_pos]
    calix_1_frame['NEG'] -= data_frame[cal_2_neg]
    calix_1_frame['POL'] -= data_frame[cal_2_pol]
    
    return calix_1_frame

def fully_enumerate_set(binding_file,
                        binding_file_directory,
                        prefix_list):
    """
    Takes a .csv with adsorption values, and a prefix list of calixarene names
    The prefix list **has already had the test set removed**
    Creates a half-enumerated list of inhibitor 1 / inhibitor 2 combinations
    Only 'half-enumerated' because is ('CP2', 'AP3') is an entry, then ('AP3', 'CP2')
    will be excluded (prevents leakage between training/validation set)
    

    Parameters
    ----------
    binding_file : string
        Name of .csv file that contains binding information
    binding_file_directory : string
        Name of directory that contains binding file
    prefix_list : list of strings
        List of names of calixarenes to be used in test/training set

    Returns
    -------
    Three lists:
    
    calix_pairs : an ordered list of [(calix_1, calix_2),] combinations
    value_tuple_list : an ordered list of tuples, where each entry is
                       [((calix_1 ADS_VAL, calix_1 UNCERT, calix_1, ADD/DUAL),
                         (calix_2 ADS_VAL, calix_2 UNCERT, calix_1 ADD/DUAL))]

    """
    
    csv_path = Path('.', binding_file_directory)
    
    load_frame = pd.read_csv(csv_path / binding_file,
                                   header=0,
                                   index_col=0)

    adsorption_frame = load_frame.where(pd.notnull(load_frame), None)
    
    calix_pairs = []
    value_tuple_list = []
    
    for entry in range(len(prefix_list)):
        for second_entry in range(len(prefix_list) - entry):
            calix_pairs.append((prefix_list[entry], prefix_list[(entry + second_entry)]))
            value_tuple_list.append(((adsorption_frame.at[prefix_list[entry], 'ADS_VAL'],
                                     adsorption_frame.at[prefix_list[entry], 'UNCERT'],
                                     adsorption_frame.at[prefix_list[entry], 'ADD_DUAL']),
                                    (adsorption_frame.at[prefix_list[entry + second_entry], 'ADS_VAL'],
                                     adsorption_frame.at[prefix_list[entry + second_entry], 'UNCERT'],
                                     adsorption_frame.at[prefix_list[entry + second_entry], 'ADD_DUAL'])))
    
    return calix_pairs, value_tuple_list

def key_to_tensor(inverse_flag,
                  calix_tuple,
                  data_frame):
    """
    This is the Pytorch 'itemgetter' that is called during training and/or validation
    Inverse flag is set by 'itemgetter'
        If 'False', the recorded problem is called
        If 'True', the inverse problem is generated
    Takes in a tuple of two calixarene prefixes and the DataFrame with ASO/NEG/POS/POL values
    Creates the difference frame via labelled_example_generator
    Turns that frame into a Pytorch tensor
    Does not associate and adsorption value - this was done by ordering in fully_enumerate_list    

    Parameters
    ----------
    inverse_flag : TYPE
        DESCRIPTION.
    calix_tuple : TYPE
        DESCRIPTION.
    data_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    Pytorch tensor with 3D data for processing

    """
    if inverse_flag == False:
        difference_frame = labelled_example_generator(calix_tuple[0], calix_tuple[1], data_frame)
    
    if inverse_flag == True:
        difference_frame = labelled_example_generator(calix_tuple[1], calix_tuple[0], data_frame)
        
    key_dimension = int(np.rint(np.cbrt(difference_frame.shape[0])))
    key_cols = difference_frame[['ASO', 'POS', 'NEG', 'POL']]
    numpy_ar = key_cols.values
    numpy_shaped = numpy_ar.reshape(key_dimension, key_dimension, key_dimension, 4)
    numpy_moveax = np.moveaxis(numpy_shaped, 3, 0)
    tensor = torch.from_numpy(numpy_moveax)
    tensor = tensor.float()

    return tensor

def calculate_ads_w_error(value_tuple):
    """
    A function that takes an inhibitor value tuple and returns
    a predicted adsorption difference that includes experimental error
    If error was listed in paper, function call will return a (slightly) different
    value each time

    Parameters
    ----------
    value_tuple : tuple of tuples
        A tuple of the form ((calix_1 ADS_VAL, calix_1 UNCERT, calix_1, ADD/DUAL),
                             (calix_2 ADS_VAL, calix_2 UNCERT, calix_1 ADD/DUAL))

    Returns
    -------
    A float value for the predicted relative adsorption
    Will not allow a negative value to be returned, in rare case uncertaintly drops adsorption value below zero
    """

    first_val = value_tuple[0][0]
    second_val = value_tuple[1][0]
    
    
    adsorption_value = (first_val / second_val)
    
    log_value = np.log(adsorption_value)
    
    return log_value


#Quick test

# def quick_test():
    
#     return_dict = coordinate_load(['50C_Uni_Cut15_Comb_Nor_20A.pq',],
#                                   'PQFiles')
#     valid_list = ['PSC4', 'PNO2', 'AP8', 'AO1', 'F4', 'CP1']
#     prefix_list = calixarene_list(return_dict['50C_Uni_Cut15_Comb_Nor_20A.pq'],
#                                   valid_list)
#     print(prefix_list)
#     calix_pairs, peptide_list, log_pair_values = fully_enumerate_set('ads_val_ani.csv',
#                                                                      'CSVFiles',
#                                                                      prefix_list)
#     key_to_tensor(False,
#                   ('AP1', 'AP3'),
#                   return_dict['50C_Uni_Cut15_Comb_Nor_20A.pq'])
#     key_to_tensor(True,
#                   ('CP1', 'AO2'),
#                   return_dict['50C_Uni_Cut15_Comb_Nor_20A.pq'])
#     peptide_one_hot('one_hot_short.csv',
#                     'CSVFiles',
#                     'H3K4')
    
#     return

    

