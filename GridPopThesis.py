#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:58:26 2020

@author: jvh

Top level function for generating populated grids: UVic calixarene project V2

"""
import ImportMol.IsoSmilesToStruct as ISS
import Alignment.IsoAlignTools as IAT
import GridTools.AniConfGPU as ACG
import GridTools.FillGrid as FG
import time
import pyarrow.parquet as pq

#Exclude compouds with bromine, due to limitations of torch Ani-2x
#Ignore PSC6 - only compoud in the entire family, not reasonable target
compound_dict = {'PSC4': ['core_a', ['psc4']],
                 'PNO2': ['core_no2', ['psc']],
                 'AP1': ['core_a', ['p1']],
                 'AP3': ['core_a', ['p3']],
                 'AP4': ['core_a', ['p4']],
                 'AP5': ['core_a', ['p5']],
                 'AP6': ['core_a', ['p6']],
                 'AP7': ['core_a', ['p7']],
                 'AP8': ['core_a', ['p8']],
                 'AP9': ['core_a', ['p9']],
                 'AM1': ['core_a', ['m1']],
                 'AM2': ['core_a', ['m2']],
                 'AH1': ['core_a', ['h1']],
                 'AH2': ['core_a', ['h2']],
                 'AH3': ['core_a', ['h3']],
                 'AH4': ['core_a', ['h4']],
                 'AH5': ['core_a', ['h5']],
                 'AH6': ['core_a', ['h6']],
                 'AH7': ['core_a', ['h7']],
                 'AO1': ['core_a', ['o1']],
                 'AO2': ['core_a', ['o2']],
                 'AO3': ['core_a', ['o3']],
                 'BP0': ['core_b', ['p0']],
                 'BP1': ['core_b', ['p1']],
                 'BM1': ['core_b', ['m1']],
                 'BH2': ['core_b', ['h2']],
                 'DP2': ['core_d', ['p2']],
                 'DM1': ['core_d', ['m1']],
                 'DO2': ['core_d', ['o2']],
                 'DO3': ['core_d', ['o3']],
                 'CP1': ['core_c', ['p1']],
                 'CP2': ['core_c', ['p2']],
                 'E1': ['core_e', ['e1']],
                 'E3': ['core_e', ['e3']],
                 'E6': ['core_e', ['e6']],
                 'E7': ['core_e', ['e7']],
                 'E8': ['core_e', ['e8']],
                 'E9': ['core_e', ['e9']],
                 'E11': ['core_e', ['e11']],
                 'F2': ['core_f', ['h1']],
                 'F3': ['core_f', ['p5']],
                 'F4': ['core_f4', ['f4']]}

setting_dict = {'spacing': 1.0,
                'size': 15.0,
                'type': 'boltzmann',
                'num_conf': 50,
                'force_level': 0.01,
                'list_factor': 2}

output_name = 'AlokThesis10A_'

mlddec_model = ISS.load_mlddec(78)
ani2x_model = ACG.load_ani2()

def create_shallow_grids(compound_dict,
                         setting_dict,
                         output_name):
    """
    A script that creates and saves populated .pq files with
    4 channels (ASO, POL, POS, NEG)

    Parameters
    ----------
    compound_dict : dictionary
        A dictionary of compound grids to create
        Each key is the name the compound will be saves with, with each
        entry being a list of parameters necessary to run
        ISS.construct_and_minimize_single
        Actual entry is a list with [core, side_chains]

    setting_dict : dictionary
        A dictionary of grid settings (such as spacing, width, etc)
        that will provide the necessary information to create
        filled shallow grids (Shallow = only 4 channels)
        Requires keys to be 'spacing', 'size', 'type', 'num_conf', 'force_level', 'list_factor'
            Spacing = distance between grid points in Å
            Size = half-width of grid in angstroms (15A means x/y/z go from -15 to +15)
            Type = 'uniform' or 'boltzmann', refers to type of conformer ranking
            Num_conf = Number of conformers to generate per calixarene
            Force_level = Amount of residual force allowed during minimization. A previous test
            of force level vs. accuracy vs. time to minimize revealed 0.01 is sufficient and efficient
            List_factor = for 'uniform' filling, this is the energy cutoff, for 'boltzmann' filling,
            this is the Boltzmann compression factor.

    Returns
    -------
    None - but filled grids are saved as .pq files

    """
    
    for dict_key in compound_dict:
        working_calix, working_energy_list = ISS.construct_and_minimize_single('CSVFiles',
                                                                               'ThesisIsoCore.csv',
                                                                               compound_dict[dict_key][0],
                                                                               compound_dict[dict_key][1],
                                                                               setting_dict['num_conf'],
                                                                               setting_dict['force_level'],
                                                                               mlddec_model,
                                                                               ani2x_model)
        if setting_dict['type'] == 'uniform':
            uniform_list = FG.create_uniform_list(working_energy_list,
                                                  setting_dict['list_factor'])
            working_frame = FG.fill_shallow_grid(setting_dict['size'],
                                                 setting_dict['spacing'],
                                                 working_calix,
                                                 uniform_list)
        elif setting_dict['type'] == 'boltzmann':
            boltzmann_list = FG.create_boltzmann_list(working_energy_list,
                                                      setting_dict['list_factor'])
            working_frame = FG.fill_shallow_grid(setting_dict['size'],
                                                 setting_dict['spacing'],
                                                 working_calix,
                                                 boltzmann_list)
        else:
            raise ValueError('Grid filling type not identified as "uniform" or "boltzmann"')
        
        save_string = output_name + dict_key + '.pq'
        
        working_frame.to_parquet(save_string)
    
    return

create_shallow_grids(compound_dict,
                     setting_dict,
                     output_name)                                                              
        
        
        
        
        
        
        
        
        