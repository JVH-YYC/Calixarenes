"""
Nearly empty file that just imports all the main functions for testing in interactive mode.
"""

import ResNet.CalixNet as RCN
import DataLoaders.CDKDataLoader as CDL

model_translation_dict = {'AP4': ['/home/jvh/Desktop/Trained Calix ResNets/', 'Initial LOO AP4_iter_0.pt'],
                          'BH2': ['/home/jvh/Desktop/Trained Calix ResNets/', 'Initial LOO BH2_iter_0.pt'],
                          'CP1': ['/home/jvh/Desktop/Trained Calix ResNets/', 'Initial LOO CP1_iter_0.pt']}

pq_file_directory = 'PQFiles'
pq_file_name = 'AlokThesis10A_Comb.pq'
csv_file_directory = 'CSVFiles'
binding_file = 'Data excluding non-binders.csv'
one_hot_file = 'one_hot_short.csv'
exclude_calix = ['E9',]
training_batch_size = 400
input_block_list = [2,2,2,2]
dropout_amount = 0.0
absolute_training = False
absolute_predictions = True
output_name = 'first_dict_pickle_test'

RCN.compile_predicted_actual_LOO_dict(model_translation_dict=model_translation_dict,
                                      pq_file_directory=pq_file_directory,
                                      pq_file_name=pq_file_name,
                                      csv_file_directory=csv_file_directory,
                                      binding_file=binding_file,
                                      one_hot_file=one_hot_file,
                                      exclude_calix=exclude_calix,
                                      training_batch_size=training_batch_size,
                                      input_block_list=input_block_list,
                                      dropout_amount=dropout_amount,
                                      absolute_training=absolute_training,
                                      absolute_predictions=absolute_predictions,
                                      output_name=output_name)