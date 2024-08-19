#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:10:38 2020

@author: jvh

Version 2 of calixarene structure generator

Input csv have 3 columns: lables, SMILES strings, and flags for core vs side-chain


"""
import numpy as np
import pandas as pd
import mlddec
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers
from pathlib import Path
import Alignment.IsoAlignTools as IAT
import GridTools.AniConfGPU as ACG
import os


def csv_import(path_name,
               file):
    """
    Parameters
    ----------
    path_name: string
        Name of the sub-directory that contains the CSV file of interest
    file : string
        A filename for a csv file that contains SMILES strings for cores and
        sidechains of calixarenes
        All csv files are held in sub-folder named 'CSVFiles'

    Returns
    -------
    A pandas dataframe generated from the csv file, and two dictionaries
    One dictionary of cores, one of sidechains (with keys that equal csv labels)
    
    """
    
    core_dict = {}
    side_dict = {}

    csv_path = Path('.', path_name)
    return_frame = pd.read_csv(csv_path / file)
    
    for index, frame_row in return_frame.iterrows():
        if frame_row['Label'] == 'c':
            core_dict[frame_row['Name']] = frame_row['Smiles']
        elif frame_row['Label'] == 's':
            side_dict[frame_row['Name']] = frame_row['Smiles']
        else:
            print('CSV Index', index, 'not properly labelled')

    return return_frame, core_dict, side_dict

def create_calix(core,
                 core_dict,
                 side_chains,
                 side_chain_dict):
    """
    Parameters
    ----------
    core : string
        label for the core that will be used to generate a calixarene.
    core_dict: dictionary (from csv_import())
        dictionary with SMILES strings for calixarene cores. Keys are
        strings (will match 'core'), entries are SMILES strings
    side_chains : tuple
        label for the side chain that will be used to generate a calixarene.
        first entry will be 'Y' type substituents
        second entry will be 'W' type substituent if present
    side_chain_dict: dictionary (from csv_import())
        dictionary with SMILES strings for calixarene side-chains. Keys are
        string (will match 'side_chain'), entries are SMILES strings.

    Returns
    -------
    A SMILES string for the new calixarene. It will be turned into a molecule
    and initial minimization performed in initial_minimization()
    
    """    
    #Define reaction to add sidechain to cores
    y_rxn = AllChem.ReactionFromSmarts("[Y][*:1].[Y][*:2]>>[*:1][*:2]") #reaction to add sidechain to core
    w_rxn = AllChem.ReactionFromSmarts("[W][*:1].[W][*:2]>>[*:1][*:2]") # reaction for addition to phenolic oxygen
    
    calix_core = Chem.MolFromSmiles(core_dict[core], sanitize=False)
    calix_side_y = Chem.MolFromSmiles(side_chain_dict[side_chains[0]], sanitize=False)
    if len(side_chains) > 1:
        calix_side_w = Chem.MolFromSmiles(side_chain_dict[side_chains[1]])
    
    #Calculate the number of reactions needed to attach 'Y' and 'W' substituents
    y_count = 0
    w_count = 0
    for atom in calix_core.GetAtoms():
        if atom.GetSymbol() == 'Y':
            y_count = y_count + 1
        elif atom.GetSymbol() == 'W':
            w_count = w_count + 1
            
    #Do first reaction, iterate (#Y - 1) further reactions
    new_mol = y_rxn.RunReactants([calix_core, calix_side_y])
    
    for repeat in range(y_count - 1):
        new_mol = y_rxn.RunReactants([new_mol[0][0], calix_side_y])
    
    #Check for 'W' attachment points. If present, attach these to the second side chain
    if w_count != 0:
        new_mol = w_rxn.RunReactants([new_mol[0][0], calix_side_w])
    
    for repeat in range(w_count - 1):
        new_mol = w_rxn.RunReactants([new_mol[0][0], calix_side_w])
    
    #RDKit Pre-Condition Violation requires going back to SMILES
    int_smiles = Chem.MolToSmiles(new_mol[0][0])
    
    return int_smiles

def load_mlddec(dielectric):
    """
    A simple function to load the trained ML models of Riniker lab for
    assigning charges. Two possible dielectric values (4 and 78) are
    provided by the lab.

    Parameters
    ----------
    dielectric : integer
        The dielectric constant of the medium, either non-poler (4) or water (78)

    Returns
    -------
    A trained ML charge-assigning model for use in other functions

    """

    ml_model = mlddec.load_models(dielectric)
    
    return ml_model

def add_mlddec_charge(molecule,
                      mlddec_model):
    """
    A function to add charged derived from Riniker Lab's mlddec ML model

    Parameters
    ----------
    molecule : RDKit Molecule object
        The calixarene molecule that will have charges added
    mlddec_model : trained ML models
        The default ML models provided by Riniker lab. These are loaded
        outside of this function, so that they are not re-loaded when
        multiple molecules are having charges added

    Returns
    -------
    None - but molecule is changed in place.

    """    
    
    mlddec.add_charges_to_mol(molecule, mlddec_model)
    
    return

def initial_minimize(smiles_string,
                      mlddec_model):
    """
    A function that takes a 2D molecule (returned by create_calix()) and does
    an initial minimization. It then locates the deprotonated phenol (using
    detect_deprot_phenol() from AlignTools) and measures the distances to all other
    phenols (with aid of enum_phenols() from AlignTools). If the three
    closest -OH phenols are within the distance cutoff of TKTK Å, the molecule
    passes. If the minimization is unsuccessful, the procedure is done again

    Parameters
    ----------
    smiles_string : string
        A SMILES string of a calixarene, returned from create_calix()
    mlddec_model : trained ML models
        The default ML models provided by Riniker lab. These are loaded
        outside of this function, so that they are not re-loaded when
        multiple molecules are having charges added    

    Returns
    -------
    An RDKit molecule that has been minimized in MMFF, and it has been checked
    to see that all 4 phenols are on the same face of the molecule

    """
    passed_check = False
    counter = 0
    while passed_check == False:
        
        #Build molecule, embed and return first conformer
        test_mol = Chem.MolFromSmiles(smiles_string)
        test_mol = Chem.AddHs(test_mol)
        AllChem.EmbedMolecule(test_mol)
        
        ff = rdForceFieldHelpers.UFFGetMoleculeForceField(test_mol)
        first_conf = test_mol.GetConformers()[0]
        mlddec.add_charges_to_mol(test_mol, mlddec_model)
        
        full_phenol_list = IAT.all_calix_phenol(test_mol)
        deprot_idx, phenol_list = IAT.iso_deprot_phenol(test_mol,
                                                             full_phenol_list)

        for oh_phenol in phenol_list:
            ff.UFFAddDistanceConstraint(deprot_idx,
                                        oh_phenol,
                                        False,
                                        2.0,
                                        3.7,
                                        5000)
        
        success_flag = 1    
        while success_flag != 0:
            success_flag = ff.Minimize(maxIts=200)
        
        distance, distance_list = IAT.define_measure_plane(first_conf,
                                                                deprot_idx,
                                                                phenol_list)
 
        if distance < 0.5 and distance_list[2][1] < 5.0:
            passed_check = True
            print('Successful minimization after', counter, 'iterations with distance', distance)
            print('With O– ••• HO distances of:', distance_list[0][1],
                  distance_list[1][1], distance_list[2][1])
            return test_mol

        else:
            counter = counter + 1
        
        if counter > 100:
            #Flag calixarenes that didn't initially minimize to all up conformer, but don't crash
            print('Construction failed after 100 iterations on SMILES: ', smiles_string)
            print('Planar distance was', distance)
            print('With O- and ••• HO distances of:', distance_list[0][1],
                  distance_list[1][1], distance_list[2][1])
            return test_mol
        
def construct_and_minimize_single(path_name,
                                  file,
                                  core,
                                  side_chains,
                                  num_conf,
                                  force,
                                  ml_charge_model,
                                  ani2x_model):
    """
    

    Parameters
    ----------
    path_name: string
        Name of the sub-directory that contains the CSV file of interest
    file : string
        A filename for a csv file that contains SMILES strings for cores and
        sidechains of calixarenes
        All csv files are held in sub-folder named 'CSVFiles'
    core : string
        Label for the calixarene core that will be constructed
    side_chains : tuple
        label for the side chain that will be used to generate a calixarene.
        first entry will be 'Y' type substituents
        second entry will be 'W' type substituent if present    
    num_conf : integer
        the number of conformers that will be generated
    force : float
        the maximum residual force allowed during ANI-2x minimization
    ml_charge_model : trained mlddec model
        A pre-trained model from Riniker lab that assigns partial charges
    ani2x_model : trained ANI-2x model
        A pre-trained model from multi-lab collaboration for conformer energies
    
    Returns
    -------
    molecule : RDKit molecule object with embedded conformers
    energy_list: an ordered list of conformer energies for grid population

    """
    #Load csv file with SMILES descriptions of cores and side chains
    return_frame, core_dict, side_chain_dict = csv_import(path_name, file)
    
    #Create intermediate SMILES string of single calixarene
    int_smiles = create_calix(core,
                              core_dict,
                              side_chains,
                              side_chain_dict)
    
    #Create molecule from SMILES, add charges, and perform initial minimize
    new_calixarene = initial_minimize(int_smiles, ml_charge_model)
        
    #Create and minimize all conformers
    conf_energy_list = ACG.gen_min_confs_l(new_calixarene,
                                           num_conf,
                                           force,
                                           ani2x_model)
    
    #Align all conformers
    IAT.align_all_confs(new_calixarene)
        
    return new_calixarene, conf_energy_list


    
#------------------------
#       Unit Tests
#------------------------

def plane_dist_check(path_name,
                     file,
                     core,
                     side_chains,
                     iters):
    """
    A unit test to see what the results are from measuring the distance between
    a plane made up of (deprotonated phenol, closest -OH phenol, 2nd closest) and
    the point (third closest phenol)

    Parameters
    ----------
    ----------
    path_name: string
        Name of the sub-directory that contains the CSV file of interest
    file : string
        A filename for a csv file that contains SMILES strings for cores and
        sidechains of calixarenes
        All csv files are held in sub-folder named 'CSVFiles'
    core : string
        label for the core that will be used to generate a calixarene.
    side_chains : tuple
        label for the side chain that will be used to generate a calixarene.
        first entry will be 'Y' type substituents
        second entry will be 'W' type substituent if present
    iters : integer
        The number of times the calixarene will be constructed, minimized,
        and plane distance measured

    Returns
    -------
    None - but prints a list of results while running

    """
    
    return_frame, core_dict, side_dict = csv_import(path_name, file)
    int_smiles = create_calix(core,
                              core_dict,
                              side_chains,
                              side_dict)
    construct_print(int_smiles,
                    iters)

    return

def construct_print(smiles_string,
                    iters):
    """
    The sub-routine that actual generates, measures, and prints distances
    when called by plane_dist_check() see above

    Parameters
    ----------
    smiles_string : string
        A SMILES string of a calixarene, returned from create_calix()
    iters : integer
        The number of times the calixarene will be constructed, minimized,
        and plane distance measured

    Returns
    -------
    None - but prints a list of results while running

    """    

    for repeat in range(iters):
        #Build molecule, embed and return first conformer
        test_mol = Chem.MolFromSmiles(smiles_string)
        test_mol = Chem.AddHs(test_mol)
        AllChem.EmbedMolecule(test_mol)
        ff = rdForceFieldHelpers.UFFGetMoleculeForceField(test_mol)
        first_conf = test_mol.GetConformers()[0]
        
        full_phenol_list = IAT.all_calix_phenol(test_mol)
        deprot_idx, phenol_list = IAT.iso_deprot_phenol(test_mol,
                                                             full_phenol_list)

        for oh_phenol in phenol_list:
            ff.UFFAddDistanceConstraint(deprot_idx,
                                        oh_phenol,
                                        False,
                                        2.0,
                                        3.7,
                                        500)
        
        success_flag = 1    
        while success_flag != 0:
            success_flag = ff.Minimize(maxIts=200)

        #Relax distance constraints, reminimize            
        for oh_phenol in phenol_list:
            ff.UFFAddDistanceConstraint(deprot_idx,
                                        oh_phenol,
                                        False,
                                        2.0,
                                        4.5,
                                        5)
        
        success_flag = 1    
        while success_flag != 0:
            success_flag = ff.Minimize(maxIts=200)
        

        distance = IAT.define_measure_plane(first_conf,
                                                 deprot_idx,
                                                 phenol_list)
        
        print('For construction:', repeat + 1, 'distance was', distance)
    
    return
            
def first_unit_test(path_name,
                    file,
                    core,
                    side_chains):
    """
    A unit test for the first two functions: load a .csv file, take a given core
    and side chain, and construct the molecule
    
    Parameters
    ----------
    path_name: string
        Name of the sub-directory that contains the CSV file of interest
    file : string
        A filename for a csv file that contains SMILES strings for cores and
        sidechains of calixarenes
        All csv files are held in sub-folder named 'CSVFiles'
    core : string
        label for the core that will be used to generate a calixarene.
    side_chains : tuple
        label for the side chain that will be used to generate a calixarene.
        first entry will be 'Y' type substituents
        second entry will be 'W' type substituent if present

    Returns
    -------
    An RDKit Molecule object that is the constructed calixarene

    """

    return_frame, core_dict, side_dict = csv_import(path_name, file)
    int_smiles = create_calix(core,
                              core_dict,
                              side_chains,
                              side_dict)
    mlddec_model = load_mlddec(dielectric=78)
    return_mol = initial_minimize(int_smiles,
                                   mlddec_model)
    
    return return_mol

def charge_check(path_name,
                 file,
                 core,
                 side_chains,
                 dielectric):
    """
    A unit test that loads and constructs a calixarene, assigns the partial
    charges, and then computes the sum of all partial charges. These calixarenes
    have multiple anions, so if sum of charges is neutral, for example, something has
    gone wrong.

    Parameters
    ----------
    path_name: string
        Name of the sub-directory that contains the CSV file of interest
    file : string
        A filename for a csv file that contains SMILES strings for cores and
        sidechains of calixarenes
        All csv files are held in sub-folder named 'CSVFiles'
    core : string
        label for the core that will be used to generate a calixarene.
    side_chains : tuple
        label for the side chain that will be used to generate a calixarene.
        first entry will be 'Y' type substituents
        second entry will be 'W' type substituent if present
    dielectric : integer
        The dielectric to use (either 4 or 78) when assigning charges

    Returns
    -------
    None - but prints key result while running

    """

    return_frame, core_dict, side_dict = csv_import(path_name, file)
    int_smiles = create_calix(core,
                              core_dict,
                              side_chains,
                              side_dict)

    test_mol = Chem.MolFromSmiles(int_smiles)
    test_mol = Chem.AddHs(test_mol)
    AllChem.EmbedMolecule(test_mol)
    
    ml_model = mlddec.load_models(78)
    mlddec.add_charges_to_mol(test_mol, ml_model)
    
    charge_sum = 0
    for atom in test_mol.GetAtoms():
        atom_charge = float(atom.GetProp('PartialCharge'))
        charge_sum = charge_sum + atom_charge
    
    print('Total charge was:', charge_sum)
    
    return

def export_conf_to_view(path_name,
                        file,
                        core,
                        side_chains,
                        num_conf,
                        force,
                        export_name):
    """
    

    Parameters
    ----------
    path_name: string
        Name of the sub-directory that contains the CSV file of interest
    file : string
        A filename for a csv file that contains SMILES strings for cores and
        sidechains of calixarenes
        All csv files are held in sub-folder named 'CSVFiles'
    core : string
        label for the core that will be used to generate a calixarene.
    side_chains : tuple
        label for the side chain that will be used to generate a calixarene.
        first entry will be 'Y' type substituents
        second entry will be 'W' type substituent if present
    num_conf : integer
        the number of conformers that will be generated
    force : float
        the maximum residual force allowed during ANI-2x minimization
    export_name : string
        File name for describing all of the .mol files that will be exported

    Returns
    -------
    None, but saves a .mol file for every conformer generated/minimized
    These conformers can then be viewed in Maestro to ensure that nothing
    is going wrong during minimization/update of atomic positions

    """    
    
    new_calixarene, conf_energy_list = construct_and_minimize_single(path_name,
                                                                     file,
                                                                     core,
                                                                     side_chains,
                                                                     num_conf,
                                                                     force)
    
    for conf_deletion in range(len(new_calixarene.GetConformers())): 
        output_name = export_name + str(conf_deletion) + '.sdf'
        output_func = Chem.SDWriter(output_name)
        output_func.write(new_calixarene)
        output_func.close()
        new_calixarene.RemoveConformer(conf_deletion)
    
    #Update: looks good! Test batches of conformers find multiple reasonable conformations
    #with all 4 phenols on same side, but also other known calixarene conformers (3 up
    #1 down, for example)
    return
    
def export_single_conf_sdf(path_name,
                           file,
                           core,
                           side_chains,
                           num_conf,
                           force,
                           export_name):
    """
    

    Parameters
    ----------
    path_name: string
        Name of the sub-directory that contains the CSV file of interest
    file : string
        A filename for a csv file that contains SMILES strings for cores and
        sidechains of calixarenes
        All csv files are held in sub-folder named 'CSVFiles'
    core : string
        label for the core that will be used to generate a calixarene.
    side_chains : tuple
        label for the side chain that will be used to generate a calixarene.
        first entry will be 'Y' type substituents
        second entry will be 'W' type substituent if present
    num_conf : integer
        the number of conformers that will be generated
    force : float
        the maximum residual force allowed during ANI-2x minimization
    export_name : string
        File name for describing all of the .mol files that will be exported

    Returns
    -------
    None, but saves a .mol file for every conformer generated/minimized
    These conformers can then be viewed in Maestro to ensure that nothing
    is going wrong during minimization/update of atomic positions

    """    
    ml_charge_model = load_mlddec(78)
    ani2x_model = ACG.load_ani2()

    
    new_calixarene, conf_energy_list = construct_and_minimize_single(path_name,
                                                                     file,
                                                                     core,
                                                                     side_chains,
                                                                     num_conf,
                                                                     force,
                                                                     ml_charge_model,
                                                                     ani2x_model)
    
    for conf_num in range(len(new_calixarene.GetConformers())): 
        output_name = export_name + str(conf_num) + '.sdf'
        output_func = Chem.SDWriter(output_name)
        output_func.write(new_calixarene, confId=conf_num)
        output_func.close()
    
    return
    
def export_conf_in_loop(molecule,
                        energy_list,
                        output_prefix):
    """
    A function similar to above, which saves every conformer for a given molecule
    for quality control checking further on. SDF file name is appended with
    relative energy.

    Parameters
    ----------
    molecule : RDKit molecule object
        Given calixarene with embedded and minimized conformers
    energy_list : list
        List of energies for all embedded conformers
    output_prefix : string
        Name of calixarene being investigated

    Returns
    -------
    None, but saves .sdf files of all conformers embedded

    """
    
    for conf_num in range(len(molecule.GetConformers())):
        output_energy = str(energy_list[conf_num])
        if len(output_energy) > 5:
            output_energy = output_energy[1:5]
        output_name = output_prefix + '_' + output_energy + '.sdf'
        
        output_func = Chem.SDWriter(output_name)
        output_func.write(molecule, confId=conf_num)
        output_func.close()
    
    return
    
def export_conf_to_folder(molecule,
                          energy_list,
                          output_prefix,
                          folder_name):
    """
    A function similar to above, which saves every conformer for a given molecule
    for quality control checking further on. SDF file name is appended with
    relative energy.

    Parameters
    ----------
    molecule : RDKit molecule object
        Given calixarene with embedded and minimized conformers
    energy_list : list
        List of energies for all embedded conformers
    output_prefix : string
        Name of calixarene being investigated
    folder_name : string
        Top-level name of the folder to save .sdf files in    
        
    Returns
    -------
    None, but saves .sdf files of all conformers embedded

    """
    
    start_dir = os.getcwd()
    
    top_fold = os.getcwd() + '/HypoConf'
    
    for conf_num in range(len(molecule.GetConformers())):
        output_energy = str(energy_list[conf_num])
        if len(output_energy) > 5:
            output_energy = output_energy[1:5]
        output_name = output_prefix + '_' + output_energy + '.sdf'
        
        if os.path.exists(top_fold + '/' + folder_name) == False:
            os.mkdir(top_fold + '/' + folder_name)
        
        os.chdir(top_fold + '/' + folder_name)
        output_func = Chem.SDWriter(output_name)
        output_func.write(molecule, confId=conf_num)
        output_func.close()
        os.chdir(start_dir)
    
    return
    
    