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
                 input_channels):
        super(ResNet, self).__init__()
        self.working_inp_channel = 64
        self.beg_conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.beg_conv1_bn = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

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
        in_tensor = self.fc1(in_tensor)
        in_tensor = self.fc1_bn(in_tensor)
        in_tensor = self.relu(in_tensor)
        in_tensor = self.fc2(in_tensor)
        in_tensor = self.fc2_bn(in_tensor)
        in_tensor = self.relu(in_tensor)
        in_tensor = self.fc3(in_tensor)
        
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
    

def ResNet18(input_channels):
    return ResNet(ResBlock,
                  [2, 2, 2, 2],
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
                    learning_rate):
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
    
    return loss_function, optimize

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
                  training_epochs,
                  learning_rate,
                  current_iteration):
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
    
    adsorption_data = RelativeAdsorptionDataset(pq_file_directory,
                                            pq_file_name,
                                            csv_file_directory,
                                            binding_file,
                                            one_hot_file,
                                            exclude_calix,
                                            test_set,
                                            batch_size)

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
        
    loss_func, optimize = loss_and_optim(network, learning_rate)

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
            #Gather input values
            cal_tens1, cal_tens2, peptide_tens, target_values = data
            target_values = target_values.view(-1, 1)
            
            #Send data to GPU
            cal_tens1 = cal_tens1.to('cuda')
            cal_tens2 = cal_tens2.to('cuda')
            inputs = cal_tens1 - cal_tens2
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

        print("Epoch {}, training_loss: {:.2f}, took: {:.2f}s".format(epoch+1, current_loss / num_batches, time.time() - epoch_time))

        #At end of epoch, try val set on GPU
        
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                cal_tens1, cal_tens2, peptide_tens, target_values = data
                target_values = target_values.view(-1, 1)
                
                #Send data to GPU
                cal_tens1 = cal_tens1.to('cuda')
                cal_tens2 = cal_tens2.to('cuda')
                inputs = cal_tens1 - cal_tens2
                peptide_tens = peptide_tens.to('cuda')
                target_values = target_values.to('cuda')
            
                #Forward pass only
                val_output = network(inputs, peptide_tens)
                val_loss_size = loss_func(val_output, target_values)
                total_val_loss += float(val_loss_size.item())
        
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

        if current_patience == 0:
            save_string = output_name[:24] + "_iter_" + str(current_iteration) + '.pt'
            torch.save(current_best_state_dict, save_string)
            print('Iteration', str(current_iteration), 'finished')
            print('Early stopping engaged at epoch: ', epoch + 1)
            print('Model saved from epoch: ', current_best_epoch)
            print("Training finished, took {:.2f}s".format(time.time() - training_start))
            
            train_pred, train_act = single_forward_pass(network,
                                                        train_loader,
                                                        '_train',
                                                        save_string[:-3])
            
            val_pred, val_act = single_forward_pass(network,
                                                    val_loader,
                                                    '_val',
                                                    save_string[:-3])
            
            return training_log

    print('Iteration', str(current_iteration), 'finished')
    print("Training finished, took {:.2f}s".format(time.time() - training_start))
    print('Model saved from epoch: ', current_best_epoch)
    save_string = output_name[:24] + "_iter_" + str(current_iteration) + '.pt'
    torch.save(current_best_state_dict, save_string)

    train_pred, train_act = single_forward_pass(network,
                                                train_loader,
                                                '_train',
                                                save_string[:-3])
    
    val_pred, val_act = single_forward_pass(network,
                                            val_loader,
                                            '_val',
                                            save_string[:-3])
    
    test_pred, test_act = single_test_pass(network,
                                             adsorption_data,
                                             save_string[:-3])
    
    return training_log

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
                  training_epochs,
                  learning_rate,
                  current_iteration):
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
    current_iteration : integer
        Counter for keeping rack of multiple trainings w/ different hyperparameters

    Returns
    -------
    Training log for plotting, and best network is saved during training

    """
    
    network = ResNet18(4)
    
    network = network.float()
    
    training_log = train_network(network,
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
                                  training_epochs,
                                  learning_rate,
                                  current_iteration)
    
    return training_log

def batch_work_flow(file_name_variable,
                    pq_file_directory,
                    pq_file_name_list,
                    csv_file_directory,
                    binding_file,
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
        
        current_training_log = cnn_work_flow(pq_file_directory=pq_file_directory,
                                              pq_file_name=file,
                                              csv_file_directory=csv_file_directory,
                                              binding_file=binding_file,
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
            #Gather input values
            cal_tens1, cal_tens2, peptide_tens, target_values = data
            target_values = target_values.view(-1, 1)
            
            #Send data to GPU
            cal_tens1 = cal_tens1.to('cuda')
            cal_tens2 = cal_tens2.to('cuda')
            inputs = cal_tens1 - cal_tens2
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

    with torch.no_grad():
        final_predict_list = []
        final_actual_list = []
        for example in range(len(dataset_obj.test_pairs)):
            cal_tens1 = dataset_obj.test_tensor_dict[dataset_obj.test_pairs[example][0]]
            cal_tens2 = dataset_obj.test_tensor_dict[dataset_obj.test_pairs[example][1]]
            peptide_tens = dataset_obj.one_hot_tags[dataset_obj.test_peptides[example]]
            target_values = dataset_obj.test_log_vals[example]

            cal_tens1 = cal_tens1.to('cuda')
            cal_tens2 = cal_tens2.to('cuda')
            inputs = cal_tens1 - cal_tens2
            peptide_tens = peptide_tens.to('cuda')
            target_values = torch.tensor(target_values).view(-1, 1)
            target_values = target_values.to('cuda')

            this_output = network(inputs, peptide_tens)
            tensor_list = list(this_output.flatten())
            predict_list = [x.item() for x in tensor_list]
            final_predict_list = final_predict_list + predict_list
            actual_tens = list(target_values.flatten())
            actual_list = [x.item() for x in actual_tens]
            final_actual_list = final_actual_list + actual_list

    plot_act_pred(final_predict_list,
                    final_actual_list,
                    'test',
                    output_name)
    
    return final_predict_list, final_actual_list


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









    
        
        