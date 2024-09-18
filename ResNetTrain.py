#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:58:26 2020

@author: jvh

Top level function for training CNN network: UVic calixarene project V2

"""

import ResNet.CalixNet as CNN
import DataLoaders.CDKDataLoader as CDL

# Initial training of the network
# initial_calix_list = ['E11', 'BH2', 'BP0', 'AH2', 'AH1',
#                       'DM1', 'BM1', 'AM2', 'AM1', 'DP2',
#                       'CP2', 'CP1', 'BP1', 'PNO2', 'PSC4',
#                       'F4', 'AP9', 'F3', 'AP8', 'F2', 'AP7',
#                       'AP6', 'E8', 'AP5', 'E7', 'AP4', 'E6',
#                       'AP3', 'E3', 'AP1', 'E1', 'DO3', 'DO2',
#                       'AO3', 'AO2', 'AO1', 'AH7', 'AH6', 'AH5']

# for calix in initial_calix_list:
#     print('Processing calixarene:', calix)
#     pq_file_directory = 'PQFiles'
#     pq_file_name = 'AlokThesis10A_Comb.pq'
#     csv_file_directory = 'CSVFiles'
#     binding_file = 'Data excluding non-binders.csv'
#     one_hot_file = 'one_hot_short.csv'
#     exclude_calix = ['E9',]
#     test_set = [calix,]
#     output_name ='Rel HiDO ' + calix
#     batch_size = 400
#     val_split = 0.1
#     min_epochs = 100
#     training_epochs = 400
#     learning_rate = 0.00033
#     resnet_block_list = [2,2,3,4]
#     dropout_amount = 0.3
#     absolute_training = False
#     absolute_predictions = False
    # state_dict_directory = '/home/jvh/Desktop/Trained Calix ResNets/'
    # state_dict_name = 'First Inverse Training E_iter_0.pt'

    # CNN.cnn_work_flow(pq_file_directory,
    #                 pq_file_name,
    #                 csv_file_directory,
    #                 binding_file,
    #                 one_hot_file,
    #                 exclude_calix,
    #                 test_set,
    #                 output_name,
    #                 batch_size,
    #                 val_split,
    #                 min_epochs,
    #                 training_epochs,
    #                 learning_rate,
    #                 resnet_block_list,
    #                 dropout_amount,
    #                 absolute_training,
    #                 absolute_predictions,
    #                 save_model=True)

    # CNN.load_and_test_saved_model(state_dict_directory,
    #                             state_dict_name,
    #                             pq_file_directory,
    #                             pq_file_name,
    #                             csv_file_directory,
    #                             binding_file,
    #                             one_hot_file,
    #                             exclude_calix,
    #                             test_set,
    #                             output_name,
    #                             batch_size,
    #                             absolute_predictions)

#Hyperparameter tuning

num_searches = 10
pq_file_directory = 'PQFiles'
pq_file_name = 'AlokThesis10A_Comb.pq'
csv_file_directory = 'CSVFiles'
binding_file = 'Overall med cat data excluding non-binders.csv'
one_hot_file = 'one_hot_short.csv'
exclude_calix = ['E9',]
output_name = 'aCNN overallCat rd 1'
batch_size = 400
val_split = 0.1
min_epochs = 100
training_epochs = 400
learning_rate_list = [0.033, 0.01, 0.0033, 0.001, 0.00033]
lr_patience = 30
resnet_block_list = [[2, 2, 2, 2],
                     [2, 2, 3, 3],
                     [3, 3, 2, 2],
                     [3, 3, 3, 3],
                     [2, 3, 3, 2],
                     [3, 2, 2, 3]]
dropout_amount_list = [0.0, 0.025, 0.05, 0.1, 0.2,]
# Above values are fixed for round 1, below are altered for round 2
# learning_rate_list = [0.033, 0.01, 0.0033, 0.001, 0.00033]
# resnet_block_list = [[2, 2, 2, 2],
#                      [2, 2, 3, 3],
#                      [2, 2, 4, 4],
#                      [2, 2, 3, 4],
#                      [2, 2, 2, 3],
#                      [2, 3, 3, 2]]
# dropout_amount_list = [0.1, 0.15, 0.2, 0.25, 0.3]
absolute_training = True
absolute_predictions = True
save_all_models = False
save_best_model = True
classification = True

CNN.random_calix_hyper_search(num_searches=num_searches,
                              pq_file_directory=pq_file_directory,
                              pq_file_name=pq_file_name,
                              csv_file_directory=csv_file_directory,
                              binding_file=binding_file,
                              one_hot_file=one_hot_file,
                              exclude_calix=exclude_calix,
                              output_name=output_name,
                              batch_size=batch_size,
                              val_split=val_split,
                              min_epochs=min_epochs,
                              training_epochs=training_epochs,
                              learning_rate_list=learning_rate_list,
                              lr_patience=lr_patience,
                              dropout_amount_list=dropout_amount_list,
                              resnet_block_list=resnet_block_list,
                              absolute_training=absolute_training,
                              absolute_predictions=absolute_predictions,
                              save_all_models=save_all_models,
                              save_best_model=save_best_model,
                              classification=classification)
                    




