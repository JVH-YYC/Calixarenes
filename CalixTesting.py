"""
Nearly empty file that just imports all the main functions for testing in interactive mode.
"""

import ResNet.CalixNet as RCN
import DataLoaders.CDKDataLoader as CDL

model_translation_dict = {'AP4': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO AP4.pt'],
                          'AH1': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO AH1.pt'],
                          'AH2': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO AH2.pt'],
                          'AH5': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO AH5.pt'],
                          'AH6': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO AH6.pt'],
                          'AH7': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                    'Rel HiDO AH7.pt'],
                          'AM1': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO AM1.pt'],
                          'AM2': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO AM2.pt'],
                          'AO1': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO AO1.pt'],
                          'AO2': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO AO2.pt'],
                          'AO3': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO AO3.pt'],
                          'AP1': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO AP1.pt'],
                          'AP3': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO AP3.pt'],
                          'AP5': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO AP5.pt'],
                          'AP6': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO AP6.pt'],
                          'AP7': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO AP7.pt'],
                          'AP8': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO AP8.pt'],
                          'AP9': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO AP9.pt'],
                          'BH2': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO BH2.pt'],
                          'BM1': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO BM1.pt'],
                          'BP0': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO BP0.pt'],
                          'BP1': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO BP1.pt'],
                          'CP1': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO CP1.pt'],
                          'CP2': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO CP2.pt'],
                          'DM1': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO DM1.pt'],
                          'DO2': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO DO2.pt'],
                          'DO3': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO DO3.pt'],
                          'DP2': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO DP2.pt'],
                          'E1': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                 'Rel HiDO E1.pt'],
                          'E3': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                 'Rel HiDO E3.pt'],
                          'E6': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                 'Rel HiDO E6.pt'],
                          'E7': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                 'Rel HiDO E7.pt'],
                          'E8': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                 'Rel HiDO E8.pt'],                        
                          'E11': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                  'Rel HiDO E11.pt'],
                          'F2': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                 'Rel HiDO F2.pt'],
                          'F3': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                 'Rel HiDO F3.pt'],
                          'F4': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                 'Rel HiDO F4.pt'],
                          'PNO2': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                   'Rel HiDO PNO2.pt'],
                          'PSC4': ['/home/jvh/Desktop/Trained Calix ResNets/High DO Rd 2/',
                                   'Rel HiDO PSC4.pt']}
pq_file_directory = 'PQFiles'
pq_file_name = 'AlokThesis10A_Comb.pq'
csv_file_directory = 'CSVFiles'
binding_file = 'Data excluding non-binders.csv'
one_hot_file = 'one_hot_short.csv'
exclude_calix = ['E9',]
training_batch_size = 400
input_block_list = [2,2,3,4]
dropout_amount = 0.3
absolute_training = False
absolute_predictions = True
output_name = 'HiDo Rd2 LOO'

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