#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:58:26 2020

@author: jvh

Top level function for training CNN network: UVic calixarene project V2

"""

import ResNet.CalixNet as CNN
import DataLoaders.CDKDataLoader as CDL

initial_calix_list = ['E11', 'BH2', 'BP0', 'AH2', 'AH1',
                      'DM1', 'BM1', 'AM2', 'AM1', 'DP2',
                      'CP2', 'CP1', 'BP1', 'PNO2', 'PSC4',
                      'F4', 'AP9', 'F3', 'AP8', 'F2', 'AP7',
                      'AP6', 'E8', 'AP5', 'E7', 'AP4', 'E6',
                      'AP3', 'E3', 'AP1', 'E1', 'DO3', 'DO2',
                      'AO3', 'AO2', 'AO1', 'AH7', 'AH6', 'AH5']

for calix in initial_calix_list:
    pq_file_directory = 'PQFiles'
    pq_file_name = 'Alok_Thesis_Comb_10A.pq'
    csv_file_directory = 'CSVFiles'
    binding_file = 'Data excluding non-binders.csv'
    one_hot_file = 'one_hot_short.csv'
    exclude_calix = ['BH5', 'E9', 'F1', 'E2', 'E5']
    test_set = [calix,]
    output_name ='Initial LOO ' + calix
    batch_size = 400
    val_split = 0.1
    training_epochs = 300
    learning_rate = 0.003
    current_iteration = 0

    CNN.cnn_work_flow(pq_file_directory,
                    pq_file_name,
                    csv_file_directory,
                    binding_file,
                    one_hot_file,
                    exclude_calix,
                    test_set,
                    output_name,
                    batch_size,
                    val_split,
                    training_epochs,
                    learning_rate,
                    current_iteration)

# CNN.batch_work_flow(file_name_variable=False,
#                     pq_file_directory='PQFiles',
#                     pq_file_name_list=['50C_Boltz_Fac2_shallow_Comb_Nor.pq',],
#                     csv_file_directory='CSVTrain',
#                     binding_file='Design_Val_NoErr.csv',
#                     test_set=['p1_8',
#                               'p1_8_alt',
#                               'p1_21',
#                               'p1_21_alt',
#                               'p1_38',
#                               'p1_38_alt',
#                               'p2_21',
#                               'p2_40',
#                               'p3_4aq',
#                               'p3_4bh',
#                               'p4_6',
#                               'p5_9',
#                               'p6_10',
#                               'p6_27',],
#                     output_name='Feb20_NoErr',
#                     batch_size_variable=False,
#                     batch_size_list=[80,],
#                     val_split=0.2,
#                     training_epochs=300,
#                     learn_rate_variable=False,
#                     learn_rate_list=[0.003,])

                                     
                    
                    




