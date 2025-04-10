"""
Nearly empty file that just imports all the main functions for testing in interactive mode.
"""

import ResNet.CalixNet as RCN
import DataLoaders.CDKDataLoader as CDL

model_translation_dict = {'AH3': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AH3.pt'],
                          'AP4': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AP4.pt'],
                          'AH1': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AH1.pt'],
                          'AH2': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AH2.pt'],
                           'AH4': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AH4.pt'],
                           'AH5': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AH5.pt'],
                          'AH6': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AH6.pt'],
                          'AH7': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                    'Small Abs LOO CNN AH7.pt'],
                          'AM1': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AM1.pt'],
                          'AM2': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AM2.pt'],
                          'AO1': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AO1.pt'],
                          'AO2': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AO2.pt'],
                          'AO3': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AO3.pt'],
                          'AP1': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AP1.pt'],
                          'AP3': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AP3.pt'],
                          'AP5': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AP5.pt'],
                          'AP6': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AP6.pt'],
                          'AP7': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AP7.pt'],
                          'AP8': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AP8.pt'],
                          'AP9': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN AP9.pt'],
                          'E1': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                 'Small Abs LOO CNN E1.pt'],
                          'E3': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                 'Small Abs LOO CNN E3.pt'],
                          'E6': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                 'Small Abs LOO CNN E6.pt'],
                          'E7': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                 'Small Abs LOO CNN E7.pt'],
                          'E8': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                 'Small Abs LOO CNN E8.pt'],                        
                          'E11': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                  'Small Abs LOO CNN E11.pt'],
                          'PNO2': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                   'Small Abs LOO CNN PNO2.pt'],
                          'PSC4': ['/home/jvh/Desktop/Trained Calix ResNets/Small Set Absolute/',
                                   'Small Abs LOO CNN PSC4.pt'],}
pq_file_directory = 'PQFiles'
pq_file_name = 'AlokThesis10A_Comb.pq'
csv_file_directory = 'CSVFiles'
binding_file = 'Small calix set for CNN.csv'
one_hot_file = 'one_hot_short.csv'
exclude_calix = ['E9', 'F2', 'F3', 'F4','BP0', 'BP1', 'BH2', 'BM1', 'CP1',
                 'CP2', 'DP2', 'DM1', 'DO2', 'DO3']
training_batch_size = 400
input_block_list = [3,3,3,3]
dropout_amount = 0.04
absolute_training = True
absolute_predictions = True
output_name = 'Small Abs LOO CNN'
classification = False

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
                                      output_name=output_name,
                                      classification=classification)