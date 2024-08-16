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
import GridTools.CDKAlignTools as CAT
import GridTools.CDKConfGPU as CCG

def core_import(csv_directory,
               core_csv):
    """
    Parameters
    ----------
    csv_directory: string
        Name of the sub-directory that contains the CSV file of interest
    core_csv : string
        A filename for a csv file that contains SMILES strings for heterocycle
        cores used in CHEM599 project

    Returns
    -------
    A pandas dataframes, with the translation between core labels 'C1' and SMILES strings
    
    """
    

    csv_path = Path('.', csv_directory)
    
    load_core = pd.read_csv(csv_path / core_csv, index_col = 0)
    core_frame = load_core.where(pd.notnull(load_core), None)
        
    return core_frame

def create_heterocycle(core,
                       core_frame,
                       cmpd_u_chain,
                       cmpd_v_chain,
                       cmpd_w_chain,
                       cmpd_y_chain,
                       cmpd_k_chain):
    """
    Parameters
    ----------
    core : string
        label for the core that will be used to generate a heterocycle for testing or training
    core_frame: pandas Dataframe from core_import
        Dataframe that translates core label 'C1' into a SMILES string
    cmpd_u_chain : string
        A SMILES string, with a Uranium atom showing position of attachment to core
    cmpd_v_chain : string
        A SMILES string with a Vanadium atom showing position of attachment to core
    cmpd_w_chain : string
        A SMILES string with a Tungsten atom showing position of attachment to core
    cmpd_y_chain : string
        A SMILES string with an Yttrium atom showing position of attachment to core
    cmpd_k_chain : string
        A SMILES string with a Potassium atom showing position of attachment to core
        

    Returns
    -------
    A SMILES string for the new heterocycle. It will be turned into a molecule
    and initial minimization performed in initial_minimization()
    
    """    
    #Define all needed reactions to construct molecules
    u_attach = AllChem.ReactionFromSmarts('[U][*:1].[U][*:2]>>[*:1][*:2]')
    v_attach = AllChem.ReactionFromSmarts('[V][*:1].[V][*:2]>>[*:1][*:2]')
    w_attach = AllChem.ReactionFromSmarts('[W][*:1].[W][*:2]>>[*:1][*:2]')
    y_attach = AllChem.ReactionFromSmarts('[Y][*:1].[Y][*:2]>>[*:1][*:2]')
    k_attach = AllChem.ReactionFromSmarts('[K][*:1].[K][*:2]>>[*:1][*:2]')
    
    u_cyclize = AllChem.ReactionFromSmarts('([*:1][U].[U][*:2])>>[*:1][*:2]')
    v_cyclize = AllChem.ReactionFromSmarts('([*:1][V].[V][*:2])>>[*:1][*:2]')
    w_cyclize = AllChem.ReactionFromSmarts('([*:1][W].[W][*:2])>>[*:1][*:2]')
    y_cyclize = AllChem.ReactionFromSmarts('([*:1][Y].[Y][*:2])>>[*:1][*:2]')
    k_cyclize = AllChem.ReactionFromSmarts('([*:1][K].[K][*:2])>>[*:1][*:2]')
    
    
    
    #Seems as though cyclization is possible using same attachment rxn (test to be safe) 
    
    working_mol = Chem.MolFromSmiles(core_frame.loc[core, 'CORE_SMILES'], sanitize=False)
    reacted_mol = None
    
    if cmpd_u_chain is not None:
        u_chain_mol = Chem.MolFromSmiles(cmpd_u_chain, sanitize=False)
        reacted_mol = u_attach.RunReactants([working_mol, u_chain_mol])
    
    if cmpd_v_chain is not None:
        v_chain_mol = Chem.MolFromSmiles(cmpd_v_chain, sanitize=False)
        if reacted_mol is not None:
            reacted_mol = v_attach.RunReactants([reacted_mol[0][0], v_chain_mol])
        else:
            reacted_mol = v_attach.RunReactants([working_mol, v_chain_mol])
            
    if cmpd_w_chain is not None:
        w_chain_mol = Chem.MolFromSmiles(cmpd_w_chain, sanitize=False)
        if reacted_mol is not None:
            reacted_mol = w_attach.RunReactants([reacted_mol[0][0], w_chain_mol])
        else:
            reacted_mol = w_attach.RunReactants([working_mol, w_chain_mol])
    
    if cmpd_y_chain is not None:
        y_chain_mol = Chem.MolFromSmiles(cmpd_y_chain, sanitize=False)
        if reacted_mol is not None:
            reacted_mol = y_attach.RunReactants([reacted_mol[0][0], y_chain_mol])
        else:
            reacted_mol = y_attach.RunReactants([working_mol, y_chain_mol])
    
    if cmpd_k_chain is not None:
        k_chain_mol = Chem.MolFromSmiles(cmpd_k_chain, sanitize=False)
        if reacted_mol is not None:
            reacted_mol = k_attach.RunReactants([reacted_mol[0][0], k_chain_mol])
        else:
            reacted_mol = k_attach.RunReactants([working_mol, k_chain_mol])
    
    #Check for need for cyclization; cyclize if necessary
    current_smiles = Chem.MolToSmiles(reacted_mol[0][0])
    if 'U' in current_smiles:
        reacted_mol = u_cyclize.RunReactants([reacted_mol[0][0],])
    if 'V' in current_smiles:
        reacted_mol = v_cyclize.RunReactants([reacted_mol[0][0],])
    if 'W' in current_smiles:
        reacted_mol = w_cyclize.RunReactants([reacted_mol[0][0],])
    if 'Y' in current_smiles:
        reacted_mol = y_cyclize.RunReactants([reacted_mol[0][0],])
    if 'K' in current_smiles:
        reacted_mol = k_cyclize.RunReactants([reacted_mol[0][0],])
        
    intermediate_smiles = Chem.MolToSmiles(reacted_mol[0][0], allHsExplicit=True)

    return intermediate_smiles    

#---------------------------------------
#       Create Heterocycle Test
#---------------------------------------

# core_frame = core_import('CSVTrain',
#                           'CHEM599_cores.csv')

# # Test C1 (pyridine) with full side-chain complement
# test_smiles1 = create_heterocycle('C1',
#                                   core_frame,
#                                   '[U][H]',
#                                   '[V][H]',
#                                   'FC1=CC(OC)=C([W])C=C1',
#                                   '[Y][H]',
#                                   'O=C([K])[C@H]1CCC[C@@H](NC(C)=O)C1')
# # passed 

# # Test both isotope labelling of roscovitine to see if cyclization works
# test_smiles2 = create_heterocycle('C3',
#                                   core_frame,
#                                   '[U]NCC1=CC=CC=C1',
#                                   '[V]/N=C\\N(C(C)C)[W].C',
#                                   None,
#                                   None,
#                                   'CC[C@@H]([K])CO')
# test_smiles3 = create_heterocycle('C4',
#                                   core_frame,
#                                   'CC[C@@H](N[U])CO',
#                                   None,
#                                   '[Y]/N=C\\N(C(C)C)[W].C',
#                                   None,
#                                   '[K]CC1=CC=CC=C1')
# # passed

def molecule_set_to_csv(csv_directory,
                        core_csv,
                        molecule_set_csv,
                        output_name):
    """
    A function that takes a core molecule directory, and a .csv of actual
    molecules, and saves a new .csv file that has the molecule described as
    only a SMILES string

    Parameters
    ----------
    csv_directory: string
        Name of the sub-directory that contains the CSV file of interest
    core_csv : string
        A filename for a csv file that contains SMILES strings for heterocycle
        cores used in CHEM599 project
    molecule_set_csv : string
        A filename for a csv file that contains the core and side-chain SMILES
        strings for constructing training/student submission molecules
    output_name : string
        String that will be used for the filename that is used to create a .csv
        file that has full molecules SMILES strings, rather than a description
        as Core/Chain lists

    Returns
    -------
    None, but saves the list of SMILES strings as a .csv file

    """
    core_frame = core_import(csv_directory,
                             core_csv)
    
    mol_path = Path('.', csv_directory)
    mol_load = pd.read_csv(mol_path / molecule_set_csv, index_col=0)
    mol_frame = mol_load.where(pd.notnull(mol_load), None)

    mol_dict = {}

    for mol_name, data_row in mol_frame.iterrows():
        try:
            print(data_row['CORE'])
            mol_smiles = create_heterocycle(data_row['CORE'],
                                            core_frame,
                                            data_row['U_CHAIN'],
                                            data_row['V_CHAIN'],
                                            data_row['W_CHAIN'],
                                            data_row['Y_CHAIN'],
                                            data_row['K_CHAIN'])
            
            mol_dict[mol_name] = mol_smiles
        except:
            print('Failure on molecule', mol_name)

    #Create and save .csv file of created molecules
    full_smiles_frame = pd.DataFrame.from_dict(mol_dict, orient='index', columns=['SMILES',])
    full_smiles_frame.index.name='Molecule Label'
    save_path = Path('.', output_name)
    full_smiles_frame.to_csv(save_path)
    
    return
    
def submission_check(csv_directory,
                     core_csv,
                     submission_csv):
    """
    A function that builds the molecule corresponding to every row in a submission .csv,
    and flags rows that have failed to manual review.

    Parameters
    ----------
    csv_directory: string
        Name of the sub-directory that contains the CSV file of interest
    core_csv : string
        A filename for a csv file that contains SMILES strings for heterocycle
        cores used in CHEM599 project
    submission_csv : string
        A filename for a csv file that contains the core and side-chain SMILES
        strings for constructing training/student submission molecules

    Returns
    -------
    Nothing - but checks every entry in the submission CSV file to ensure
    that all molecules can be properly constructed.

    """

    core_frame = core_import(csv_directory,
                             core_csv)
    
    sub_path = Path('.', csv_directory)
    submit_load = pd.read_csv(sub_path / submission_csv, index_col=0)
    submit_frame = submit_load.where(pd.notnull(submit_load), None)

    for mol_name, data_row in submit_frame.iterrows():
        try:
            print(data_row['CORE'])
            create_heterocycle(data_row['CORE'],
                               core_frame,
                               data_row['U_CHAIN'],
                               data_row['V_CHAIN'],
                               data_row['W_CHAIN'],
                               data_row['Y_CHAIN'],
                               data_row['K_CHAIN'])
        except:
            print('Failure on molecule', mol_name)
    
    return

    

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
        The heterocycle molecule that will have charges added
    mlddec_model : trained ML models
        The default ML models provided by Riniker lab. These are loaded
        outside of this function, so that they are not re-loaded when
        multiple molecules are having charges added

    Returns
    -------
    None - but molecule is changed in place.

    """    
    
    mlddec.add_charges_to_mol(molecule,
                              mlddec_model)
    
    return

def initial_minimize(smiles_string,
                      mlddec_model):
    """
    A function that takes a 2D molecule (returned by create_heterocycle()) and does
    an initial minimization. 

    Parameters
    ----------
    smiles_string : string
        A SMILES string of an aminoheterocycle, returned from create_heterocycle()
    mlddec_model : trained ML models
        The default ML models provided by Riniker lab. These are loaded
        outside of this function, so that they are not re-loaded when
        multiple molecules are having charges added    

    Returns
    -------
    An RDKit molecule that has been minimized in UFF
    (will be minimzed by torchani later)

    """
        
    #Build molecule, embed and return first conformer
    heterocycle_mol = Chem.MolFromSmiles(smiles_string)
    heterocycle_mol = Chem.AddHs(heterocycle_mol)
    AllChem.EmbedMolecule(heterocycle_mol)
    
    ff = rdForceFieldHelpers.UFFGetMoleculeForceField(heterocycle_mol)
    ff.Minimize(maxIts=200)
    mlddec.add_charges_to_mol(heterocycle_mol,
                              mlddec_model)
    
    return heterocycle_mol

        
def construct_and_minimize_single(cmpd_name,
                                  num_conf,
                                  force,
                                  ml_charge_model,
                                  ani2x_model,
                                  csv_directory,
                                  core_csv,
                                  cmpd_core,
                                  cmpd_u_chain=None,
                                  cmpd_v_chain=None,
                                  cmpd_w_chain=None,
                                  cmpd_y_chain=None,
                                  cmpd_k_chain=None,
                                  flip_flag=False):
    """
    

    Parameters
    ----------
    cmpd_name : string
        Given name of molecule from first column of Pandas dataframe
    num_conf : integer
        the number of conformers that will be generated
    force : float
        the maximum residual force allowed during ANI-2x minimization
    ml_charge_model : trained mlddec model
        A pre-trained model from Riniker lab that assigns partial charges
    ani2x_model : trained ANI-2x model
        A pre-trained model from multi-lab collaboration for conformer energies
    csv_directory : string
        Directory where CSV file describing heterocycle cores and specific molecules
    core_csv : string
        Name of file with core information
    cmpd_core : string
        Name of the heterocycle core being built (e.g. 'C1', 'C4')
    cmpd_u_chain : string
        A SMILES string, with a Uranium atom showing position of attachment to core
    cmpd_v_chain : string
        A SMILES string with a Vanadium atom showing position of attachment to core
    cmpd_w_chain : string
        A SMILES string with a Tungsten atom showing position of attachment to core
    cmpd_y_chain : string
        A SMILES string with an Yttrium atom showing position of attachment to core
    cmpd_k_chain : string
        A SMILES string with a Potassium atom showing position of attachment to core
    flip_flag : Boolean
        Described whether to orient the 15N labelled nitrogen atom in the heterocycle
        ring in the positive (False) or negative (True) y-direction.
    
    Returns
    -------
    molecule : RDKit molecule object with embedded conformers
    energy_list: an ordered list of conformer energies for grid population

    """
    #Load csv file with SMILES descriptions of cores
    core_frame = core_import(csv_directory,
                             core_csv)
    
    #Create proper SMILES string of single substituted heterocycle
    int_smiles = create_heterocycle(cmpd_core,
                                    core_frame,
                                    cmpd_u_chain,
                                    cmpd_v_chain,
                                    cmpd_w_chain,
                                    cmpd_y_chain,
                                    cmpd_k_chain)
    
    #Create molecule from SMILES, add charges, and perform initial minimize
    new_heterocycle = initial_minimize(int_smiles,
                                      ml_charge_model)
    #Create and minimize all conformers
    conf_energy_list = CCG.gen_min_confs_l(new_heterocycle,
                                           num_conf,
                                           force,
                                           ani2x_model)
    
    #Align all conformers
    CAT.align_all_confs(new_heterocycle,
                        flip_flag)
    
    return new_heterocycle, conf_energy_list


    
# #------------------------
# #       Unit Tests
# #------------------------

# def plane_dist_check(path_name,
#                      file,
#                      core,
#                      side_chains,
#                      iters):
#     """
#     A unit test to see what the results are from measuring the distance between
#     a plane made up of (deprotonated phenol, closest -OH phenol, 2nd closest) and
#     the point (third closest phenol)

#     Parameters
#     ----------
#     ----------
#     path_name: string
#         Name of the sub-directory that contains the CSV file of interest
#     file : string
#         A filename for a csv file that contains SMILES strings for cores and
#         sidechains of calixarenes
#         All csv files are held in sub-folder named 'CSVFiles'
#     core : string
#         label for the core that will be used to generate a calixarene.
#     side_chains : tuple
#         label for the side chain that will be used to generate a calixarene.
#         first entry will be 'Y' type substituents
#         second entry will be 'W' type substituent if present
#     iters : integer
#         The number of times the calixarene will be constructed, minimized,
#         and plane distance measured

#     Returns
#     -------
#     None - but prints a list of results while running

#     """
    
#     return_frame, core_dict, side_dict = csv_import(path_name, file)
#     int_smiles = create_calix(core,
#                               core_dict,
#                               side_chains,
#                               side_dict)
#     construct_print(int_smiles,
#                     iters)

#     return

# def construct_print(smiles_string,
#                     iters):
#     """
#     The sub-routine that actual generates, measures, and prints distances
#     when called by plane_dist_check() see above

#     Parameters
#     ----------
#     smiles_string : string
#         A SMILES string of a calixarene, returned from create_calix()
#     iters : integer
#         The number of times the calixarene will be constructed, minimized,
#         and plane distance measured

#     Returns
#     -------
#     None - but prints a list of results while running

#     """    

#     for repeat in range(iters):
#         #Build molecule, embed and return first conformer
#         test_mol = Chem.MolFromSmiles(smiles_string)
#         test_mol = Chem.AddHs(test_mol)
#         AllChem.EmbedMolecule(test_mol)
#         ff = rdForceFieldHelpers.UFFGetMoleculeForceField(test_mol)
#         first_conf = test_mol.GetConformers()[0]
        
#         full_phenol_list = IAT.all_calix_phenol(test_mol)
#         deprot_idx, phenol_list = IAT.iso_deprot_phenol(test_mol,
#                                                              full_phenol_list)

#         for oh_phenol in phenol_list:
#             ff.UFFAddDistanceConstraint(deprot_idx,
#                                         oh_phenol,
#                                         False,
#                                         2.0,
#                                         3.7,
#                                         500)
        
#         success_flag = 1    
#         while success_flag != 0:
#             success_flag = ff.Minimize(maxIts=200)

#         #Relax distance constraints, reminimize            
#         for oh_phenol in phenol_list:
#             ff.UFFAddDistanceConstraint(deprot_idx,
#                                         oh_phenol,
#                                         False,
#                                         2.0,
#                                         4.5,
#                                         5)
        
#         success_flag = 1    
#         while success_flag != 0:
#             success_flag = ff.Minimize(maxIts=200)
        

#         distance = IAT.define_measure_plane(first_conf,
#                                                  deprot_idx,
#                                                  phenol_list)
        
#         print('For construction:', repeat + 1, 'distance was', distance)
    
#     return
            
# def first_unit_test(path_name,
#                     file,
#                     core,
#                     side_chains):
#     """
#     A unit test for the first two functions: load a .csv file, take a given core
#     and side chain, and construct the molecule
    
#     Parameters
#     ----------
#     path_name: string
#         Name of the sub-directory that contains the CSV file of interest
#     file : string
#         A filename for a csv file that contains SMILES strings for cores and
#         sidechains of calixarenes
#         All csv files are held in sub-folder named 'CSVFiles'
#     core : string
#         label for the core that will be used to generate a calixarene.
#     side_chains : tuple
#         label for the side chain that will be used to generate a calixarene.
#         first entry will be 'Y' type substituents
#         second entry will be 'W' type substituent if present

#     Returns
#     -------
#     An RDKit Molecule object that is the constructed calixarene

#     """

#     return_frame, core_dict, side_dict = csv_import(path_name, file)
#     int_smiles = create_calix(core,
#                               core_dict,
#                               side_chains,
#                               side_dict)
#     mlddec_model = load_mlddec(dielectric=78)
#     return_mol = initial_minimize(int_smiles,
#                                    mlddec_model)
    
#     return return_mol

# def charge_check(path_name,
#                  file,
#                  core,
#                  side_chains,
#                  dielectric):
#     """
#     A unit test that loads and constructs a calixarene, assigns the partial
#     charges, and then computes the sum of all partial charges. These calixarenes
#     have multiple anions, so if sum of charges is neutral, for example, something has
#     gone wrong.

#     Parameters
#     ----------
#     path_name: string
#         Name of the sub-directory that contains the CSV file of interest
#     file : string
#         A filename for a csv file that contains SMILES strings for cores and
#         sidechains of calixarenes
#         All csv files are held in sub-folder named 'CSVFiles'
#     core : string
#         label for the core that will be used to generate a calixarene.
#     side_chains : tuple
#         label for the side chain that will be used to generate a calixarene.
#         first entry will be 'Y' type substituents
#         second entry will be 'W' type substituent if present
#     dielectric : integer
#         The dielectric to use (either 4 or 78) when assigning charges

#     Returns
#     -------
#     None - but prints key result while running

#     """

#     return_frame, core_dict, side_dict = csv_import(path_name, file)
#     int_smiles = create_calix(core,
#                               core_dict,
#                               side_chains,
#                               side_dict)

#     test_mol = Chem.MolFromSmiles(int_smiles)
#     test_mol = Chem.AddHs(test_mol)
#     AllChem.EmbedMolecule(test_mol)
    
#     ml_model = mlddec.load_models(78)
#     mlddec.add_charges_to_mol(test_mol, ml_model)
    
#     charge_sum = 0
#     for atom in test_mol.GetAtoms():
#         atom_charge = float(atom.GetProp('PartialCharge'))
#         charge_sum = charge_sum + atom_charge
    
#     print('Total charge was:', charge_sum)
    
#     return

# def export_conf_to_view(path_name,
#                         file,
#                         core,
#                         side_chains,
#                         num_conf,
#                         force,
#                         export_name):
#     """
    

#     Parameters
#     ----------
#     path_name: string
#         Name of the sub-directory that contains the CSV file of interest
#     file : string
#         A filename for a csv file that contains SMILES strings for cores and
#         sidechains of calixarenes
#         All csv files are held in sub-folder named 'CSVFiles'
#     core : string
#         label for the core that will be used to generate a calixarene.
#     side_chains : tuple
#         label for the side chain that will be used to generate a calixarene.
#         first entry will be 'Y' type substituents
#         second entry will be 'W' type substituent if present
#     num_conf : integer
#         the number of conformers that will be generated
#     force : float
#         the maximum residual force allowed during ANI-2x minimization
#     export_name : string
#         File name for describing all of the .mol files that will be exported

#     Returns
#     -------
#     None, but saves a .mol file for every conformer generated/minimized
#     These conformers can then be viewed in Maestro to ensure that nothing
#     is going wrong during minimization/update of atomic positions

#     """    
    
#     new_calixarene, conf_energy_list = construct_and_minimize_single(path_name,
#                                                                      file,
#                                                                      core,
#                                                                      side_chains,
#                                                                      num_conf,
#                                                                      force)
    
#     for conf_deletion in range(len(new_calixarene.GetConformers())): 
#         output_name = export_name + str(conf_deletion) + '.sdf'
#         output_func = Chem.SDWriter(output_name)
#         output_func.write(new_calixarene)
#         output_func.close()
#         new_calixarene.RemoveConformer(conf_deletion)
    
#     #Update: looks good! Test batches of conformers find multiple reasonable conformations
#     #with all 4 phenols on same side, but also other known calixarene conformers (3 up
#     #1 down, for example)
#     return
    
    
