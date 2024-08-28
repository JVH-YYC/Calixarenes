#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:34:38 2020

@author: jvh

One sided (+ inversion) data loader, CNN with starting layer geometries

DataLoader creates each calixarene tensor once and populates in dictionary. Send tensors to GPU
before substraction for more efficient training.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler
import pandas as pd
import time
import matplotlib.pyplot as plt
import random
import numpy as np
from pathlib import Path
import DataLoaders.CDKDataLoader as CDL
import pickle


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 ident_downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv1_bn = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.ident_downsample = ident_downsample
        
    def forward(self, in_tensor):
        identity = in_tensor
        
        in_tensor = self.conv1(in_tensor)
        in_tensor = self.conv1_bn(in_tensor)
        in_tensor = self.relu(in_tensor)
        in_tensor = self.conv2(in_tensor)
        in_tensor = self.conv2_bn(in_tensor)
        
        if self.ident_downsample is not None:
            identity = self.ident_downsample(identity)
        
        in_tensor = in_tensor + identity
        in_tensor = self.relu(in_tensor)
        
        return in_tensor

class ResNet(nn.Module):
    def __init__(self,
                 res_block,
                 blocks_per_layer,
                 dropout,
                 input_channels):
        super(ResNet, self).__init__()
        self.working_inp_channel = 64
        self.beg_conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.beg_conv1_bn = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(dropout)

        #ResNet type layers
        self.res_layer1 = self.create_block(ResBlock, blocks_per_layer[0], out_channels=64, stride=2)
        self.res_layer2 = self.create_block(ResBlock, blocks_per_layer[1], out_channels=128, stride=2)
        self.res_layer3 = self.create_block(ResBlock, blocks_per_layer[2], out_channels=256, stride=2)
        self.res_layer4 = self.create_block(ResBlock, blocks_per_layer[3], out_channels=512, stride=2)

        #Initial input is 48 points along each dimension, narrows down to 1x1x1 by end of layer 4
        #Peptide tensor is length 19
        self.fc1 = nn.Linear(531, 531)
        self.fc1_bn = nn.BatchNorm1d(531)
        self.fc2 = nn.Linear(531, 531)
        self.fc2_bn = nn.BatchNorm1d(531)
        self.fc3 = nn.Linear(531, 1)

    def forward(self, in_tensor, peptide_tensor):
        in_tensor = self.beg_conv1(in_tensor)
        in_tensor = self.beg_conv1_bn(in_tensor)
        in_tensor = self.relu(in_tensor)
        in_tensor = self.maxpool(in_tensor)
        
        in_tensor = self.res_layer1(in_tensor)
        in_tensor = self.res_layer2(in_tensor)
        in_tensor = self.res_layer3(in_tensor)
        in_tensor = self.res_layer4(in_tensor)
        
        in_tensor = in_tensor.reshape(in_tensor.shape[0], -1)
        in_tensor = torch.cat((in_tensor, peptide_tensor), dim=1)
        in_tensor = self.dropout(self.fc1(in_tensor))
        in_tensor = self.fc1_bn(in_tensor)
        in_tensor = self.relu(in_tensor)
        in_tensor = self.dropout(self.fc2(in_tensor))
        in_tensor = self.fc2_bn(in_tensor)
        in_tensor = self.relu(in_tensor)
        in_tensor = self.dropout(self.fc3(in_tensor))
        
        return in_tensor

    def create_block(self,
                     res_block,
                     repeat_blocks,
                     out_channels,
                     stride):        
        
        blocks=[]
        ident_downsample = None        

        if stride != 1 or self.working_inp_channel != out_channels:
            ident_downsample = nn.Sequential(nn.Conv3d(self.working_inp_channel,
                                                       out_channels,
                                                       kernel_size=1,
                                                       stride=stride),
                                             nn.BatchNorm3d(out_channels))
        blocks.append(ResBlock(self.working_inp_channel,
                               out_channels,
                               stride,
                               ident_downsample))
        self.working_inp_channel = out_channels
        
        for add_block in range(repeat_blocks - 1):
            blocks.append(ResBlock(self.working_inp_channel,
                                   out_channels))
        
        return nn.Sequential(*blocks)
    

def ResNet18(input_block_list,
             dropout_amount,
             input_channels=4):
    return ResNet(ResBlock,
                  input_block_list,
                  dropout_amount,
                  input_channels)

        
class RelativeAdsorptionDataset(Dataset):
    """ 
    Takes a .csv file with adsorption data
    Creates a fully enumerated data set via FullyEnumerateSet (returns lists of cdk_pair tuple + log_values)
    Sets a batch size for training epochs
    Getitem returns a tensor of the difference frame, and the target value
    """

    def __init__(self,
                 pq_file_directory,
                 pq_file_name,
                 csv_file_directory,
                 binding_file,
                 one_hot_file,
                 exclude_calix,
                 test_set,
                 training_batch_size):
        """
        Creates and organizes the caliarene pair/peptide/relative adsorption dataset
        Getitem returns a tensor of the difference frame, one hot encoded peptide, and target value

        Parameters
        ----------
        pq_file_directory : string
            Name of directory holding parquet grid files
        pq_file_name : strings
            Name of file that contains gridded data. 
        csv_file_directory : string
            Name of directory holding  adsorption data .csv files
        binding_file : string
            Name of csv file holding adsorption data
        one_hot_file : string
            Name of csv file holding one-hot encoded peptide data
        exclude_calix : list of strings
            Names of calixarenes to be held out of all sets - necessary as more calixarenes were calculated and
            included in .pq file than have good adsorption data
        test_set : list of strings
            Names of the calixarene files to be held out of the test/training set for validation
        training_batch_size : integer
            Size of batches for training

        Returns
        -------
        Creates training/validation data set (but not test), with necessary .len and .getitem functions

        """
        self.pq_file_directory = pq_file_directory
        self.pq_file_name = pq_file_name
        self.csv_file_directory = csv_file_directory
        self.binding_file = binding_file
        self.test_set = test_set
        self.training_batch_size = training_batch_size
        self.exclude_calix = exclude_calix
                
        self.molecule_frame = CDL.coordinate_load(pq_file_name,
                                               pq_file_directory)
        self.prefix_list = CDL.calixarene_list(self.molecule_frame,
                                               test_set + exclude_calix)
        self.calix_pairs, self.peptide_list, self.log_val_list = CDL.fully_enumerate_set(binding_file,
                                                                     csv_file_directory,
                                                                     self.prefix_list)
        
        self.absolute_ads_val = CDL.load_absolute_adsorption(csv_file_directory,
                                                             binding_file)

        self.test_pairs, self.test_peptides, self.test_log_vals = CDL.enumerate_test_calix(binding_file,
                                                                                           csv_file_directory,
                                                                                           self.prefix_list,
                                                                                           test_set)

        self.tensor_dict = CDL.create_tensor_dict(self.prefix_list,
                                                  self.molecule_frame)
        
        self.test_tensor_dict = CDL.create_tensor_dict(self.test_set + self.prefix_list,
                                                       self.molecule_frame)
        
        self.one_hot_tags = CDL.load_peptide_one_hot(csv_file_directory,
                                                     one_hot_file)

    def __len__(self):
        return len(self.calix_pairs)
    
    def __getitem__(self, idx):
    #Should return 3D data difference frame, adsorption difference
    #Use a random True/False flag to select either the forward or inverse problem randomly

        calix_pairs = self.calix_pairs[idx]

        first_tens = self.tensor_dict[calix_pairs[0]]
        second_tens = self.tensor_dict[calix_pairs[1]]

        sample_value = self.log_val_list[idx]

        peptide_tensor = self.one_hot_tags[self.peptide_list[idx]]

        return first_tens, second_tens, peptide_tensor, sample_value

class AbsoluteAdsorptionDataset(Dataset):
    """ 
    Takes a .csv file with adsorption data
    Creates a simple dataset of calixarene/peptide/values - aka absolute adsorption values
    as opposed to relative adsorption values as investigated above
    """

    def __init__(self,
                 pq_file_directory,
                 pq_file_name,
                 csv_file_directory,
                 binding_file,
                 one_hot_file,
                 exclude_calix,
                 test_set,
                 training_batch_size):
        """
        Creates and organizes the caliarene pair/peptide/relative adsorption dataset
        Getitem returns a tensor of the difference frame, one hot encoded peptide, and target value

        Parameters
        ----------
        pq_file_directory : string
            Name of directory holding parquet grid files
        pq_file_name : strings
            Name of file that contains gridded data. 
        csv_file_directory : string
            Name of directory holding  adsorption data .csv files
        binding_file : string
            Name of csv file holding adsorption data
        one_hot_file : string
            Name of csv file holding one-hot encoded peptide data
        exclude_calix : list of strings
            Names of calixarenes to be held out of all sets - necessary as more calixarenes were calculated and
            included in .pq file than have good adsorption data
        test_set : list of strings
            Names of the calixarene files to be held out of the test/training set for validation
        training_batch_size : integer
            Size of batches for training

        Returns
        -------
        Creates training/validation data set (but not test), with necessary .len and .getitem functions

        """
        self.pq_file_directory = pq_file_directory
        self.pq_file_name = pq_file_name
        self.csv_file_directory = csv_file_directory
        self.binding_file = binding_file
        self.test_set = test_set
        self.training_batch_size = training_batch_size
        self.exclude_calix = exclude_calix
                
        self.molecule_frame = CDL.coordinate_load(pq_file_name,
                                               pq_file_directory)
        self.prefix_list = CDL.calixarene_list(self.molecule_frame,
                                               test_set + exclude_calix)
        self.train_calix, self.peptide_list, self.log_val_list = CDL.simple_enumerate_set(binding_file,
                                                                     csv_file_directory,
                                                                     self.prefix_list)
        
        self.absolute_ads_val = CDL.load_absolute_adsorption(csv_file_directory,
                                                             binding_file)

        self.test_calix, self.test_peptides, self.test_log_vals = CDL.enumerate_absolute_test_calix(binding_file,
                                                                                           csv_file_directory,
                                                                                           test_set)

        self.tensor_dict = CDL.create_tensor_dict(self.prefix_list,
                                                  self.molecule_frame)
        
        self.test_tensor_dict = CDL.create_tensor_dict(self.test_set,
                                                       self.molecule_frame)
        
        self.one_hot_tags = CDL.load_peptide_one_hot(csv_file_directory,
                                                     one_hot_file)

    def __len__(self):
        return len(self.train_calix)
    
    def __getitem__(self, idx):
    #Should return 3D data difference frame, adsorption difference
    #Use a random True/False flag to select either the forward or inverse problem randomly

        active_calix = self.train_calix[idx]

        calix_tens = self.tensor_dict[active_calix]

        sample_value = self.log_val_list[idx]

        peptide_tensor = self.one_hot_tags[self.peptide_list[idx]]

        return calix_tens, peptide_tensor, sample_value

def load_trained_model(state_dict_directory,
                       state_dict_name,
                       input_block_list,
                       dropout_amount,
                       device='cuda'):
    """
    Function to re-load a previously trained model from a state dictionary file
    At this point, all models are trained as ResNet18
    Returns the model with the desired state dictionary loaded
    """

    model = ResNet18(input_block_list,
                     dropout_amount)
    model = model.float()
    state_dict = torch.load(state_dict_directory + state_dict_name, map_location=device)

    if any(key.startswith('module') for key in state_dict.keys()):
        model = nn.DataParallel(model)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model

def dataset_leakage_check(dataset_object):
    """
    A function that looks at all calixarene names found in every example in the training set
    These are printed, and compared against the calixarenes in the test set. If any overlaps are found,
    those examples are printed as well.

    Only done on relative training (functions for absolute sets were made with copy/paste + minor mods)
    """

    all_calix = set()
    test_calix = set(dataset_object.test_set)
    
    for example in range(len(dataset_object.calix_pairs)):
        all_calix.add(dataset_object.calix_pairs[example][0])
        all_calix.add(dataset_object.calix_pairs[example][1])
    
    print('All calixarenes found in training set:')
    print(all_calix)
    print('Calixarenes found in test set:')
    print(test_calix)
    
    for example in range(len(dataset_object.calix_pairs)):
        if dataset_object.calix_pairs[example][0] in test_calix:
            print('Leakage found in example:', example)
            print('Calixarene 1:', dataset_object.calix_pairs[example][0])
            print('Calixarene 2:', dataset_object.calix_pairs[example][1])
            print('Peptide:', dataset_object.peptide_list[example])
            print('Log Value:', dataset_object.log_val_list[example])
    
    return

def val_train_indices(dataset,
                       val_split,
                       shuffle=True):
    """
    Looks at length of dataset, shuffles indices (if shuffle=True), splits indices according to
    test_split.
    Returns lists of test and train indices to create data loaders.
    """
    set_length = dataset.__len__()
    index_values = list(range(set_length))
    cutoff = int(np.floor(val_split * set_length))
    
    if shuffle == True:
        np.random.shuffle(index_values)
    
    train_index, val_index = index_values[cutoff:], index_values[:cutoff]
    
    return train_index, val_index


def loss_and_optim(network,
                    learning_rate,
                    lr_patience=30):
    """
    Creates Pytorch loss function and optimizer

    Parameters
    ----------
    network : ResNet object
        ResNet convolutional network generates as described above
    learning_rate : float
        Learning rate for training, typically around 0.0001 for adam optimizer

    Returns
    -------
    Loss function and optimizer.

    """
    
    loss_function = nn.MSELoss()

    optimize = optim.Adam(network.parameters(),
                          lr=learning_rate)
    
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimize,
                                                    mode='min',
                                                    factor=0.1,
                                                    patience=lr_patience,
                                                    threshold=0.0001,
                                                    min_lr=(learning_rate / 10001),
                                                    verbose=True)


    return loss_function, optimize, lr_sched

def extract_predicted_actual(nested_dict):
    """
    Traverse a nested dictionary and extract 'predicted' and 'actual' values into corresponding lists.
    
    Args:
    - nested_dict (dict): The nested dictionary with two levels of keys. At the bottom level, there are 
                          'predicted' and 'actual' entries, each pointing to a float value.
    
    Returns:
    - predicted_list (list): A list of all 'predicted' float values, ordered.
    - actual_list (list): A list of all 'actual' float values, ordered.
    """
    predicted_list = []
    actual_list = []

    for first_key in nested_dict:
        for second_key in nested_dict[first_key]:
            data_dict = nested_dict[first_key][second_key]
            predicted_list.append(data_dict['predicted'])
            actual_list.append(data_dict['actual'])

    return predicted_list, actual_list

def train_network(network,
                  pq_file_directory,
                  pq_file_name,
                  csv_file_directory,
                  binding_file,
                  one_hot_file,
                  exclude_calix,
                  test_set,
                  output_name,
                  batch_size,
                  val_split,
                  min_epochs,
                  training_epochs,
                  learning_rate,
                  absolute_training,
                  absolute_predictions,
                  save_model,
                  save_test_dictionary):
    """
    Training loop for network training, including early stopping, data logging,
    and figure generation for review.

    Parameters
    ----------
    network : CNNet object as created above
        Convolutional network to be trained
    pq_file_directory : string
        Name of the directory holding parquet files of molecular grids
    pq_file_name : string
        Name of parquet file containing gridded molecular data
    csv_file_directory : string
        Name of folder holding CSV files
    binding_file : string
        Name of csv file with binding data
    test_set : list of strings
        Name of inhibitors to remove from training/validation set for final testing
    output_name : string
        Name that will be used to label all output files
    batch_size : integer
        Size of batch for training
    val_split : float
        Float between 0-1, amount of training set to use for validation
    training_epochs : integer
        Number of training epochs
    learning_rate : float
        Learning rate for adam optimizer
    current_iteration : integer
        Counter for use when training over multiple hyperparameters

    Returns
    -------
    Returns the recrod of training epoch vs. loss and saves .png of predicted vs actual at end of training

    """    
    if absolute_training == False:
        adsorption_data = RelativeAdsorptionDataset(pq_file_directory,
                                                pq_file_name,
                                                csv_file_directory,
                                                binding_file,
                                                one_hot_file,
                                                exclude_calix,
                                                test_set,
                                                batch_size)
    else:
        adsorption_data = AbsoluteAdsorptionDataset(pq_file_directory,
                                                pq_file_name,
                                                csv_file_directory,
                                                binding_file,
                                                one_hot_file,
                                                exclude_calix,
                                                test_set,
                                                batch_size)

    # Perform once per batch run, hash out if not needed
    # dataset_leakage_check(adsorption_data)

    train_index, val_index = val_train_indices(adsorption_data,
                                               val_split)    

    train_sample = SubsetRandomSampler(train_index)
    val_sample = SubsetRandomSampler(val_index)    

        
    train_loader = torch.utils.data.DataLoader(adsorption_data,
                                                batch_size=batch_size,
                                                sampler=train_sample)

    val_loader = torch.utils.data.DataLoader(adsorption_data,
                                              batch_size=batch_size,
                                              sampler=val_sample)

    num_batches = len(train_loader)
        
    loss_func, optimize, lr_sched = loss_and_optim(network, learning_rate)

    training_start = time.time()

    training_log = []

    best_val_loss = 1000000.0
    current_patience = 50
    current_best_epoch = 0
    
    if torch.cuda.device_count() > 1:
        print('Multiple GPU Detected')
        network = nn.DataParallel(network)
    
    network.to('cuda')
        
    for epoch in range(training_epochs):
        current_loss = 0.0
        epoch_time = time.time()

        for data in train_loader:
            #Gather input values for training type
            if absolute_training == False:
                cal_tens1, cal_tens2, peptide_tens, target_values = data
                target_values = target_values.view(-1, 1)
                
                #Send data to GPU
                cal_tens1 = cal_tens1.to('cuda')
                cal_tens2 = cal_tens2.to('cuda')
                inputs = cal_tens1 - cal_tens2
                peptide_tens = peptide_tens.to('cuda')
                target_values = target_values.to('cuda')
            else:
                inputs, peptide_tens, target_values = data
                target_values = target_values.view(-1, 1)

                #Send data to GPU
                inputs = inputs.to('cuda')
                peptide_tens = peptide_tens.to('cuda')
                target_values = target_values.to('cuda')
            
            #Set current gradients to zero
            optimize.zero_grad()
            
            #Forward, backward, optimize
            output = network(inputs, peptide_tens)
            loss_amount = loss_func(output, target_values.float())
            loss_amount.float()
            loss_amount.backward()
            optimize.step()
            
            #Update totals
            current_loss += float(loss_amount.item())
        if epoch % 10 == 0:
            print("Epoch {}, training_loss: {:.2f}, took: {:.2f}s".format(epoch+1, current_loss / num_batches, time.time() - epoch_time))

        #At end of epoch, try val set on GPU
        
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                #Gather input values for training type
                if absolute_training ==  False:
                    cal_tens1, cal_tens2, peptide_tens, target_values = data
                    target_values = target_values.view(-1, 1)
                    
                    #Send data to GPU
                    cal_tens1 = cal_tens1.to('cuda')
                    cal_tens2 = cal_tens2.to('cuda')
                    inputs = cal_tens1 - cal_tens2
                    peptide_tens = peptide_tens.to('cuda')
                    target_values = target_values.to('cuda')
                else:
                    inputs, peptide_tens, target_values = data
                    target_values = target_values.view(-1, 1)
                    
                    #Send data to GPU
                    inputs = inputs.to('cuda')
                    peptide_tens = peptide_tens.to('cuda')
                    target_values = target_values.to('cuda')
            
                #Forward pass only
                val_output = network(inputs, peptide_tens)
                val_loss_size = loss_func(val_output, target_values)
                total_val_loss += float(val_loss_size.item())
            if epoch % 10 == 0:
                print("Val loss = {:.2f}".format(total_val_loss / len(val_loader)))
            this_epoch_result = (epoch + 1, (total_val_loss / len(val_loader)))
            training_log.append(this_epoch_result)
        
        #Early stopping test
        
        if (total_val_loss / len(val_loader)) <= best_val_loss:
            best_val_loss = (total_val_loss / len(val_loader))
            current_best_state_dict = network.state_dict()
            current_best_epoch = epoch+1
            current_patience = 50
        else:
            current_patience = current_patience - 1

        if current_patience == 0 and epoch > min_epochs:
            save_string = output_name[:24] + '.pt'
            if save_model == True:
                torch.save(current_best_state_dict, save_string)
            print('Iteration finished')
            print('Early stopping engaged at epoch: ', epoch + 1)
            print('Model saved from epoch: ', current_best_epoch)
            print("Training finished, took {:.2f}s".format(time.time() - training_start))
            
            train_pred, train_act = single_forward_pass(network,
                                                        train_loader,
                                                        absolute_training,
                                                        '_train',
                                                        save_string[:-3])
            
            val_pred, val_act = single_forward_pass(network,
                                                    val_loader,
                                                    absolute_training,
                                                    '_val',
                                                    save_string[:-3])
            
            if absolute_predictions == False:
                #Not possible to do absolute training and relative predictions
                test_pred, test_act = single_test_pass(network,
                                                adsorption_data,
                                                save_string[:-3])
                test_loss = loss_func(torch.tensor(test_pred), torch.tensor(test_act))
            else:
                predict_dict = single_abs_test_pass(network,
                                                                   adsorption_data,
                                                                   absolute_training)
                
                abs_test_pred, abs_test_act = extract_predicted_actual(predict_dict)

                test_loss = loss_func(torch.tensor(abs_test_pred), torch.tensor(abs_test_act))

            return training_log, test_loss

    print('Iteration finished')
    print("Training finished, took {:.2f}s".format(time.time() - training_start))
    print('Model saved from epoch: ', current_best_epoch)
    save_string = output_name[:24] + '.pt'
    if save_model == True:
        torch.save(current_best_state_dict, save_string)

    train_pred, train_act = single_forward_pass(network,
                                                train_loader,
                                                absolute_training,
                                                '_train',
                                                save_string[:-3])
    
    val_pred, val_act = single_forward_pass(network,
                                            val_loader,
                                            absolute_training,
                                            '_val',
                                            save_string[:-3])
    
    if absolute_predictions == False:
        #Not possible to do absolute training and relative predictions
        test_pred, test_act = single_test_pass(network,
                                                adsorption_data,
                                                save_string[:-3])
        test_loss = loss_func(torch.tensor(test_pred), torch.tensor(test_act))
    else:
        test_pred_dict = single_abs_test_pass(network,
                                                           adsorption_data,
                                                           absolute_training)

        abs_test_pred, abs_test_act = extract_predicted_actual(test_pred_dict)

        test_loss = loss_func(torch.tensor(abs_test_pred), torch.tensor(abs_test_act))
    
    return training_log, test_loss

def cnn_work_flow(pq_file_directory,
                  pq_file_name,
                  csv_file_directory,
                  binding_file,
                  one_hot_file,
                  exclude_calix,
                  test_set,
                  output_name,
                  batch_size,
                  val_split,
                  min_epochs,
                  training_epochs,
                  learning_rate,
                  resnet_block_list,
                  dropout_amount,
                  absolute_training,
                  absolute_predictions,
                  save_model):
    """
    Work flow that creates network and trains it.
    Returns the training log file for plotting

    Parameters
    ----------
    pq_file_directory : string
        Directory where parquet files are found.
    pq_file_name : list of strings
        Name of files that contain gridded data. 
    csv_file_directory : string
        Name of directory holding one-hot encodings and adsorption data .csv files
    binding_file : string
        Name of csv file holding adsorption data
    one_hot_file : string
        Name of csv file holding one-hot encoded peptide data
    exclude_calix : list of strings
        List of calixarenes to be held out of all sets. This is necessary as more calixarenes were calculated and
        included in .pq file than have good adsorption data
    test_set : list of strings
        List of calixarenes that are held out for validation
    output_name : string
        Name for saving files
    batch_size : integer
        Size of batches for training
    val_split : float
        Number between 0-1 that represents what percent is reserved for test set
    training_epochs : integer
        Number of epochs for training
    learning_rate : float
        Learning rate for training, adam usually around 0.0001
    resnet_block_list : list of integers
        List of integers that represent the number of resnet blocks in each layer
    dropout_amount : float
        Amount of dropout to be used in the network
    absolute_training : boolean
        If true, will train the network on single calixarenes/absolute adsorption values, rather than
        calixarene pairs with relative absorption values
    absolute_predictions : boolean
        If true, will convert a set of relative adsorption predictions back into absolute values
    current_iteration : integer
        Counter for keeping rack of multiple trainings w/ different hyperparameters

    Returns
    -------
    Training log for plotting, and best network is saved during training

    """
    network = ResNet18(resnet_block_list,
                       dropout_amount,
                       4)
    
    network = network.float()
    
    training_log, test_loss = train_network(network,
                                  pq_file_directory,
                                  pq_file_name,
                                  csv_file_directory,
                                  binding_file,
                                  one_hot_file,
                                  exclude_calix,
                                  test_set,
                                  output_name,
                                  batch_size,
                                  val_split,
                                  min_epochs,
                                  training_epochs,
                                  learning_rate,
                                  absolute_training,
                                  absolute_predictions,
                                  save_model)
    
    return training_log, network, test_loss

def batch_work_flow(file_name_variable,
                    pq_file_directory,
                    pq_file_name_list,
                    csv_file_directory,
                    binding_file,
                    one_hot_file,
                    test_set,
                    output_name,
                    batch_size_variable,
                    batch_size_list,
                    val_split,
                    training_epochs,
                    learn_rate_variable,
                    learn_rate_list):
    """
    Batch training of networks to allow for hyperparameter searching and plotting of results

    Parameters
    ----------
    file_name_variable : TYPE
        DESCRIPTION.
    pq_file_directory : TYPE
        DESCRIPTION.
    file_name_list : TYPE
        DESCRIPTION.
    csv_file_directory : TYPE
        DESCRIPTION.
    onehot_file : TYPE
        DESCRIPTION.
    binding_file : TYPE
        DESCRIPTION.
    validation_set : TYPE
        DESCRIPTION.
    output_name : TYPE
        DESCRIPTION.
    batch_size_variable : TYPE
        DESCRIPTION.
    batch_size_list : TYPE
        DESCRIPTION.
    filter_list_variable : TYPE
        DESCRIPTION.
    filter_list : TYPE
        DESCRIPTION.
    test_split : TYPE
        DESCRIPTION.
    training_epochs : TYPE
        DESCRIPTION.
    learn_rate_variable : TYPE
        DESCRIPTION.
    learn_rate_list : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    total_experiments = max([len(pq_file_name_list),
                              len(batch_size_list),
                              len(learn_rate_list)])
    
    if file_name_variable == False:
        pq_file_name_list = pq_file_name_list * (total_experiments // len(pq_file_name_list))
    
    if batch_size_variable == False:
        batch_size_list = batch_size_list * (total_experiments // len(batch_size_list))
        
    if learn_rate_variable == False:
        learn_rate_list = learn_rate_list * (total_experiments // len(learn_rate_list))
    
    training_log_dict = {}
    current_iteration = 1
    
    for file, batch_size, learn_rate in zip(pq_file_name_list, batch_size_list, learn_rate_list):
        current_output_name = output_name + '_file=' + str(file) + '_batchs=' + str(batch_size) + '_lr=' + str(learn_rate) 
        
        print("Current settings in NoErrNet are: " + current_output_name)
        
        current_training_log, test_loss = cnn_work_flow(pq_file_directory=pq_file_directory,
                                              pq_file_name=file,
                                              csv_file_directory=csv_file_directory,
                                              binding_file=binding_file,
                                              one_hot_file=one_hot_file,
                                              test_set=test_set,
                                              output_name=current_output_name,
                                              batch_size=batch_size,
                                              val_split=val_split,
                                              training_epochs=training_epochs,
                                              learning_rate=learn_rate,
                                              current_iteration=current_iteration)
        training_log_dict[current_output_name] = current_training_log
        current_iteration = current_iteration + 1

    plot_all_results(training_log_dict,
                      output_name)
    
    return

def random_calix_hyper_search(num_searches,
                              pq_file_directory,
                              pq_file_name,
                              csv_file_directory,
                              binding_file,
                              one_hot_file,
                              exclude_calix,
                              output_name,
                              batch_size,
                              val_split,
                              min_epochs,
                              training_epochs,
                              learning_rate_list,
                              dropout_amount_list,
                              resnet_block_list,
                              absolute_training,
                              absolute_predictions,
                              save_all_models,
                              save_best_model):
    """
    Function that randomly selects hyperparameters for training a network
    """

    a_calix = ['AP1', 'AP3', 'AP4', 'AP5', 'AP6', 'AP7',
               'AP8', 'AP9', 'AM1', 'AM2', 'AH1', 'AH2',
               'AH5', 'AH6', 'AH7', 'AO1', 'AO2', 'AO3']
    b_calix = ['BP0', 'BP1', 'BM1', 'BH2']
    c_calix = ['CP1', 'CP2']
    d_calix = ['DP2', 'DM1', 'DO2', 'DO3']
    e_calix = ['E1', 'E3', 'E6', 'E7', 'E8', 'E11']
    f_calix = ['F2', 'F3', 'F4']

    training_log_dict = {}
    current_iteration = 0
    best_test_loss = 1000000000.0
    search_value_list = []

    for search in range(num_searches):
        current_iteration = current_iteration + 1
        current_output_name = output_name + '_iter_' + str(current_iteration)
        good_start = False
        test_calix_list = [random.choice(a_calix),
                           random.choice(b_calix),
                           random.choice(c_calix),
                           random.choice(d_calix),
                           random.choice(e_calix),
                           random.choice(f_calix)]
        while good_start == False:
            current_lr = random.choice(learning_rate_list)
            curr_resnet = random.choice(resnet_block_list)
            curr_dropout = random.choice(dropout_amount_list)
            curr_search_tup = (current_lr, curr_resnet, curr_dropout)
            if curr_search_tup not in search_value_list:
                search_value_list.append(curr_search_tup)
                good_start = True
        print('Good start with hyperparameters:', curr_search_tup)
        current_training_log, current_model, current_test_loss = cnn_work_flow(pq_file_directory=pq_file_directory,
                                                pq_file_name=pq_file_name,
                                                csv_file_directory=csv_file_directory,
                                                binding_file=binding_file,
                                                one_hot_file=one_hot_file,
                                                test_set=test_calix_list,
                                                exclude_calix=exclude_calix,
                                                output_name=current_output_name,
                                                batch_size=batch_size,
                                                val_split=val_split,
                                                min_epochs=min_epochs,
                                                training_epochs=training_epochs,
                                                learning_rate=current_lr,
                                                resnet_block_list=curr_resnet,
                                                dropout_amount=curr_dropout,
                                                absolute_training=absolute_training,
                                                absolute_predictions=absolute_predictions,
                                                save_model=save_all_models)
        training_log_dict[current_output_name] = current_training_log
        print('Iteration finished with test loss of:', current_test_loss)

        #Check to see if this training log has minimum error on test set. If so,
        #save the model, the list of hyperparameters, and the test loss.
        if current_test_loss < best_test_loss:
            best_output_name = current_output_name
            best_test_loss = current_test_loss
            best_hyperparameters = curr_search_tup
            best_test_calix = test_calix_list
            best_model = current_model
    
    #Save the best model to file, and write a .txt file with the best hyperparameters
    #and the test loss
    if save_best_model == True:
        save_string = best_output_name + '.pt'
        torch.save(best_model, save_string)

    with open(best_output_name + '.txt', 'w') as f:
        f.write('Best hyperparameters: ' + str(best_hyperparameters) + '\n')
        f.write('Test loss: ' + str(best_test_loss) + '\n')
        f.write('Test calixarenes: ' + str(best_test_calix) + '\n')

    plot_all_results(training_log_dict,
                        output_name)
    
    return
        
def plot_all_results(training_log_dict,
                      output_name):
    """
    Takes a dictionary that contains all of the file variables tested as dictionary key
    with a list of tuples of (epoch #, test loss).
    Convert into a plot of epoch (x) vs loss (y) with each entry as a different line

    Parameters
    ----------
    training_log_dict : dictionary
        Output from training generated in the workflow
    output_name : string
        Will become file name for .png

    Returns
    -------
    None, but saves .png file

    """        

    fig, ax = plt.subplots(figsize=(10,3))
    
    for run_name, training_log in training_log_dict.items():
        epoch_num, loss_amt = zip(*training_log)
        ax.plot(epoch_num, loss_amt, label=run_name)
        
    ax.set(xlabel='Epoch', ylabel='Test loss',
            title=output_name)
    
    fig.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                ncol=1, mode='expand', borderaxespad=0.)
    
    save_name = output_name + '.png'
    
    fig.savefig(save_name, bbox_inches='tight')

    return

def single_forward_pass(network,
                        data_loader,
                        absolute_training,
                        figure_title,
                        output_name):
    """
    Takes a trained (/early stopped) network and runs one forward pass to
    create a predicted/actual plot

    Parameters
    ----------
    model : trained Pytorch neural network
        The network being trained that will be evaluated
    data_loader : pytorch DataLoader
        Either training or validation DataLoader
    figure_title : string
        Either 'training' or 'validation' as set above
    output_name : string
        

    Returns
    -------
    None, but saves predicted vs actual plot

    """

    with torch.no_grad():
        final_predict_list = []
        final_actual_list = []
        for data in data_loader:
            #Gather input values for training type
            if absolute_training == False:
                cal_tens1, cal_tens2, peptide_tens, target_values = data
                target_values = target_values.view(-1, 1)
                
                #Send data to GPU
                cal_tens1 = cal_tens1.to('cuda')
                cal_tens2 = cal_tens2.to('cuda')
                inputs = cal_tens1 - cal_tens2
                peptide_tens = peptide_tens.to('cuda')
                target_values = target_values.to('cuda')
            else:
                inputs, peptide_tens, target_values = data
                target_values = target_values.view(-1, 1)
                
                #Send data to GPU
                inputs = inputs.to('cuda')
                peptide_tens = peptide_tens.to('cuda')
                target_values = target_values.to('cuda')
            
            #Forward pass only
            this_output = network(inputs, peptide_tens)
            tensor_list = list(this_output.flatten())
            predict_list = [x.item() for x in tensor_list]
            final_predict_list = final_predict_list + predict_list
            actual_tens = list(target_values.flatten())
            actual_list = [x.item() for x in actual_tens]
            final_actual_list = final_actual_list + actual_list
    
    plot_act_pred(final_predict_list,
                  final_actual_list,
                  figure_title,
                  output_name)
    
    return final_predict_list, final_actual_list

def single_test_pass(network,
                     dataset_obj,
                     output_name):
    """
    Takes a trained network and runs a forward pass on the test set

    Has a slightly different structure, as the test set points cannot
    be accessed by the __getitem__ function in the dataset object.

    Must be called independently for safety.
    """

    batch_size = dataset_obj.training_batch_size

    with torch.no_grad():
        final_predict_list = []
        final_actual_list = []
        
        # Initialize batch lists
        batch_cal_tens1 = []
        batch_cal_tens2 = []
        batch_peptide_tens = []
        batch_target_values = []
        
        for example in range(len(dataset_obj.test_pairs)):
            # Gather input values for this example
            cal_tens1 = dataset_obj.test_tensor_dict[dataset_obj.test_pairs[example][0]]
            cal_tens2 = dataset_obj.test_tensor_dict[dataset_obj.test_pairs[example][1]]
            peptide_tens = dataset_obj.one_hot_tags[dataset_obj.test_peptides[example]]
            target_values = dataset_obj.test_log_vals[example]
            
            # Append to batch lists
            batch_cal_tens1.append(cal_tens1)
            batch_cal_tens2.append(cal_tens2)
            batch_peptide_tens.append(peptide_tens)
            batch_target_values.append(torch.tensor(target_values).view(-1, 1))
            
            # Process the batch when it reaches the specified batch size or at the end of the dataset
            if (example + 1) % batch_size == 0 or (example + 1) == len(dataset_obj.test_pairs):
                # Stack tensors to form a batch
                batch_cal_tens1 = torch.stack(batch_cal_tens1).to('cuda')
                batch_cal_tens2 = torch.stack(batch_cal_tens2).to('cuda')
                batch_inputs = batch_cal_tens1 - batch_cal_tens2
                batch_peptide_tens = torch.stack(batch_peptide_tens).to('cuda')
                batch_target_values = torch.cat(batch_target_values).to('cuda')
                
                # Forward pass through the network
                batch_output = network(batch_inputs, batch_peptide_tens)
                
                # Convert output and targets to lists and accumulate results
                predict_list = batch_output.flatten().tolist()
                actual_list = batch_target_values.flatten().tolist()
                
                final_predict_list.extend(predict_list)
                final_actual_list.extend(actual_list)
                
                # Reset batch lists
                batch_cal_tens1 = []
                batch_cal_tens2 = []
                batch_peptide_tens = []
                batch_target_values = []

    plot_act_pred(final_predict_list,
                    final_actual_list,
                    'test',
                    output_name)
    
    return final_predict_list, final_actual_list

def single_abs_test_pass(network,
                         dataset_obj,
                         absolute_training):
    """
    Takes a trained network and runs a forward pass on the test set

    Has a slightly different structure, as the test set points cannot
    be accessed by the __getitem__ function in the dataset object.

    Must be called independently for safety.

    Predictions must be organized by peptide so that averaging can be done appropriately

    Test dataset structure for pairs is always [known_calix, test_calix]
    """

    batch_size = dataset_obj.training_batch_size

    test_return_dict = {}

    #Significant different format depending on whether training was relative or absolute
    if absolute_training == False:
        with torch.no_grad():
            running_predict_dict = {}
            running_actual_dict = {}

            #Populate dictionaries with empty lists for test hosts and all peptides
            for test_calix in dataset_obj.test_set:
                running_predict_dict[test_calix] = {}
                running_actual_dict[test_calix] = {}
                for peptide_name in dataset_obj.one_hot_tags.keys():
                    running_predict_dict[test_calix][peptide_name] = []
                    running_actual_dict[test_calix][peptide_name] = []
            
            # Initialize batch lists
            batch_cal_tens1 = []
            batch_cal1_name = []
            batch_cal_tens2 = []
            batch_cal2_name = []
            batch_peptide_tens = []
            batch_peptide_name = []
            batch_target_values = []
            batch_known_value = []

            for example in range(len(dataset_obj.test_pairs)):
                # Gather input values for this example
                cal_tens1 = dataset_obj.test_tensor_dict[dataset_obj.test_pairs[example][0]]
                cal1_name = dataset_obj.test_pairs[example][0]
                cal_tens2 = dataset_obj.test_tensor_dict[dataset_obj.test_pairs[example][1]]
                cal2_name = dataset_obj.test_pairs[example][1]
                peptide_tens = dataset_obj.one_hot_tags[dataset_obj.test_peptides[example]]
                peptide_name = dataset_obj.test_peptides[example]
                #Log absolute value is the target for prediction
                target_value = np.log(dataset_obj.absolute_ads_val.loc[cal2_name, peptide_name])
                known_value = dataset_obj.absolute_ads_val.loc[cal1_name, peptide_name]

                # Append to batch lists. Double length of named lists to account for forward and inverse prediction **inside next loop**
                batch_cal_tens1.append(cal_tens1)
                batch_cal1_name.append(cal1_name)
                batch_cal_tens2.append(cal_tens2)
                batch_cal2_name.append(cal2_name)
                batch_peptide_tens.append(peptide_tens)
                batch_peptide_name.append(peptide_name)
                batch_target_values.append(target_value)
                batch_known_value.append(known_value)

                # Process the batch when it reaches the specified batch size or at the end of the dataset
                # Include both forward and inverse prediction - must also double length of actuals
                if (example + 1) % batch_size == 0 or (example + 1) == len(dataset_obj.test_pairs):
                    batch_cal1_name = batch_cal1_name * 2
                    batch_cal2_name = batch_cal2_name * 2
                    batch_peptide_name = batch_peptide_name * 2
                    batch_target_values = batch_target_values * 2
                    batch_known_value = batch_known_value * 2
                    # Stack tensors to form a batch
                    batch_cal_tens1 = torch.stack(batch_cal_tens1).to('cuda')
                    batch_cal_tens2 = torch.stack(batch_cal_tens2).to('cuda')
                    batch_inputs = batch_cal_tens1 - batch_cal_tens2
                    batch_inverse = batch_cal_tens2 - batch_cal_tens1
                    batch_peptide_tens = torch.stack(batch_peptide_tens).to('cuda')
                    
                    # Forward pass through the network
                    batch_output = network(batch_inputs, batch_peptide_tens)
                    batch_inverse = network(batch_inverse, batch_peptide_tens)
                    # Convert output and targets to lists and accumulate results
                    predict_list = batch_output.flatten().tolist()
                    inv_predict = batch_inverse.flatten().tolist()
                    inv_predict_list = [-1 * x for x in inv_predict]
                    complete_pred_list = predict_list + inv_predict_list
                    for i in range(len(complete_pred_list)):
                        running_predict_dict[batch_cal2_name[i]][batch_peptide_name[i]].append(batch_known_value[i] / (np.exp(complete_pred_list[i])))
                        running_actual_dict[batch_cal2_name[i]][batch_peptide_name[i]].append(batch_target_values[i])


                    # Reset batch lists
                    batch_cal_tens1 = []
                    batch_cal1_name = []
                    batch_cal_tens2 = []
                    batch_cal2_name = []
                    batch_peptide_tens = []
                    batch_peptide_name = []
                    batch_target_values = []
                    batch_known_value = []
                    
        #Average all predictions for each peptide
        final_pred_dict = {}
        final_act_dict = {}

        for calix_host in running_predict_dict.keys():
            final_pred_dict[calix_host] = {}
            final_act_dict[calix_host] = {}
            for peptide in running_predict_dict[calix_host].keys():
                #Average predictions as logs for appropriate chemical sense *and return as logs to match predictions!*
                log_pred_list = [np.log(x) for x in running_predict_dict[calix_host][peptide]]
                mean_pred = np.mean(log_pred_list)
                final_pred_dict[calix_host][peptide] = mean_pred
                final_act_dict[calix_host][peptide] = np.mean(running_actual_dict[calix_host][peptide])
    else:
        with torch.no_grad():
            #No averaging is needed, as there is only 1 prediction per host/peptide combination
            final_pred_dict = {}
            final_act_dict = {}

            #Populate dictionaries with empty lists for test hosts and all peptides
            for test_calix in dataset_obj.test_set:
                final_pred_dict[test_calix] = {}
                final_act_dict[test_calix] = {}
                for peptide_name in dataset_obj.one_hot_tags.keys():
                    final_pred_dict[test_calix][peptide_name] = []
                    final_act_dict[test_calix][peptide_name] = []
            
            # Initialize batch lists
            batch_cal_tens = []
            batch_cal_name = []
            batch_peptide_tens = []
            batch_peptide_name = []
            batch_target_values = []

            for example in range(len(dataset_obj.test_calix)):
                # Gather input values for this example
                cal_tens = dataset_obj.test_tensor_dict[dataset_obj.test_calix[example]]
                cal_name = dataset_obj.test_calix[example]
                peptide_tens = dataset_obj.one_hot_tags[dataset_obj.test_peptides[example]]
                peptide_name = dataset_obj.test_peptides[example]
                target_value = dataset_obj.absolute_ads_val.loc[cal_name, peptide_name]

                # Append to batch lists
                batch_cal_tens.append(cal_tens)
                batch_cal_name.append(cal_name)
                batch_peptide_tens.append(peptide_tens)
                batch_peptide_name.append(peptide_name)
                batch_target_values.append(target_value)

                # Process the batch when it reaches the specified batch size or at the end of the dataset
                if (example + 1) % batch_size == 0 or (example + 1) == len(dataset_obj.test_calix):
                    # Stack tensors to form a batch
                    batch_cal_tens = torch.stack(batch_cal_tens).to('cuda')
                    batch_peptide_tens = torch.stack(batch_peptide_tens).to('cuda')
                    
                    # Forward pass through the network
                    batch_output = network(batch_cal_tens, batch_peptide_tens)
                    # Convert output and targets to lists and accumulate results
                    predict_list = batch_output.flatten().tolist()
                    for i in range(len(predict_list)):
                        final_pred_dict[batch_cal_name[i]][batch_peptide_name[i]].append(predict_list[i])
                        final_act_dict[batch_cal_name[i]][batch_peptide_name[i]].append(batch_target_values[i])
                    
                    # Reset batch lists
                    batch_cal_tens = []
                    batch_cal_name = []
                    batch_peptide_tens = []
                    batch_peptide_name = []
                    batch_target_values = []

    for calix_host in final_pred_dict.keys():
        test_return_dict[calix_host] = {}
        for peptide in final_pred_dict[calix_host].keys():
            test_return_dict[calix_host][peptide] = {'predicted': final_pred_dict[calix_host][peptide],
                                                      'actual': final_act_dict[calix_host][peptide]}

    return test_return_dict

def plot_act_pred(predicted_data,
                  actual_data,
                  figure_title,
                  output_name):
    """
    A function that creates a predicted vs. actual plot 

    Parameters
    ----------
    predicted_data : list
        A list of predicted values for plotting
    actual_data : list
        A list of actual results for plotting
    figure_title : string
        String indicating whether this is training or validation data
    output_name : string
        Output name as indicated from training loop 

    Returns
    -------
    None, but saves .png file

    """
    plt.figure(figsize=(10,10))
    plt.scatter(actual_data, predicted_data, c='navy', alpha=0.25)
    
    p1 = max(max(predicted_data), max(actual_data))
    p2 = min(min(predicted_data), min(predicted_data))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('Actual Values', fontsize=15)
    plt.ylabel('Predicted Values', fontsize=15)
    plt.axis('equal')

    save_string = output_name + figure_title + '.png'
    
    plt.savefig(save_string)

    return

def compile_predicted_actual_LOO_dict(model_translation_dict,
                                      pq_file_directory,
                                      pq_file_name,
                                      csv_file_directory,
                                      binding_file,
                                      one_hot_file,
                                      exclude_calix,
                                      training_batch_size,
                                      input_block_list,
                                      dropout_amount,
                                      absolute_training,
                                      absolute_predictions,
                                      output_name):
    """
    A function that takes as an input a dictionary, where the dictionary
    keys are all calixarene hosts to be investigated ('AP8', 'BM2', etc.)

    Each of these keys will point to a file directory and file name that contains
    the saved model for that specific calixarene host.

    A dataset with the target calixarene in the test set will be created, and will
    be used to make the single test pass. The results will be compiled into a final
    dictionary of the form {calixarene: {peptide: {'predicted': x, 'actual': y}}}

    The compiled dictionary will be pickled for future plotting
    """
    prediction_dict = {}

    for calix_host in model_translation_dict:
        model = load_trained_model(model_translation_dict[calix_host][0],
                                   model_translation_dict[calix_host][1],
                                   input_block_list,
                                   dropout_amount)
        if absolute_training == False:
            adsorption_data = RelativeAdsorptionDataset(pq_file_directory=pq_file_directory,
                                                pq_file_name=pq_file_name,
                                                csv_file_directory=csv_file_directory,
                                                binding_file=binding_file,
                                                one_hot_file=one_hot_file,
                                                exclude_calix=exclude_calix,
                                                test_set=[calix_host],
                                                training_batch_size=training_batch_size)
        else:
            adsorption_data = AbsoluteAdsorptionDataset(pq_file_directory=pq_file_directory,
                                                pq_file_name=pq_file_name,
                                                csv_file_directory=csv_file_directory,
                                                binding_file=binding_file,
                                                one_hot_file=one_hot_file,
                                                exclude_calix=exclude_calix,
                                                test_set=[calix_host],
                                                training_batch_size=batch_size)
            
        if absolute_predictions == True:
            ### Will iterate through the test set, as the predictions go straight into
            ### the dictionary to be returned
            curr_result_dict = single_abs_test_pass(model,
                                                    adsorption_data,
                                                    absolute_training)
            prediction_dict.update(curr_result_dict)

    #Pickle the dictionary for future plotting
    with open(output_name + '.pkl', 'wb') as f:
        pickle.dump(prediction_dict, f)

    return prediction_dict
        
        





    
        
        