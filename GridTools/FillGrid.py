#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:51:11 2020

@author: jvh

Scripts for creating and populating molecular grid, once
the molecule has been loaded, ML charge applied,
conformers generated and torchAni energy ranked

Grid filling tools needs to have the following settings:
    
    Narrow vs broad (set cut-off to 15 kcal/mol or 7.5 kcal/mol for Uniform;
                     divide by 2 or by 4 for Boltzmann distribution)
    Uniform vs Boltz (fill grids with uniform distribution, or calculate Boltz dist)
    Shallow vs Deep (are all positive charges in same layer, or are separated by atom type/hybridization)

Home computer doesn't have NVIDIA card right now - update to GPU when possible

Assumes that molecule has already been built, conformers generated, and minimized prior to calling these functions

Files are split this way because RDKit is on CPU, in time, these grid populations will be on GPU 

"""

import numpy as np
import pandas as pd
from rdkit import Chem

def create_boltzmann_list(conformer_energy_list,
                          division_factor):
    """
    A formula that calculates the population of each conformer as based on a
    Boltzmann distribution
    
    Take kbT at room temp to be 0.59 kcal/mol
    
    Fill list until 0.99999 of conformational distribution is reached.
    
    Parameters
    ----------
    conformer_energy_list : list of tuples
        A list of lists, where each entry is [relative energy, RDKit Conformer]
    division_factor : integer
        A factor that sets the amount by which energy differences are compressed if desired
    Returns
    -------
    A new list of lists, where each entry is [population fraction, RDKit Conformer]

    """

    boltz_sum = 0
    for conformer_entry in conformer_energy_list:
        boltz_sum = boltz_sum + (np.exp(((-1 / division_factor) * conformer_entry[0]) / 0.59))
    
    boltz_conf_list = []
    for conformer_entry in conformer_energy_list:
        frac_pop = (np.exp(((-1 / division_factor) * conformer_entry[0]) / 0.59)) / boltz_sum
        boltz_conf_list.append([frac_pop, conformer_entry[1]])
        
    curr_boltz_sum = 0
    
    final_boltz_list = []
    
    for sorted_conformer in boltz_conf_list:
        if curr_boltz_sum < 0.99999:
            curr_boltz_sum = curr_boltz_sum + sorted_conformer[0]
            final_boltz_list.append(sorted_conformer)
    
    
    return final_boltz_list

def create_uniform_list(conformer_energy_list,
                        cutoff_energy):
    """
    Takes an input list of conformers and energies, and given the cutoff energy,
    creates a new list that contains only conformers below that relative energy
    The conformers are evenly weighted to give a total value of 1.0

    Parameters
    ----------
    conformer_energy_list : list of lists
        A list of lists, where each entry is [relative energy, RDKit Conformer]
    cutoff_energy : float
        Relative energy in kcal/mol above which conformers are ignored

    Returns
    -------
    A new list of lists, where each entry is [population fraction, RDKit Conformer]

    """
    
    if cutoff_energy < 0:
        raise ValueError('Cut-off energy threshhold must be positive')
    
    uniform_list = []
    
    for conformer_entry in conformer_energy_list:
        if conformer_entry[0] <= cutoff_energy:
            uniform_list.append(conformer_entry)
    
    evenly_weighed_list = [[(1 / len(uniform_list)), conf_entry[1]] for conf_entry in uniform_list]
    
    return evenly_weighed_list

def generate_grid(spacing,
                  dimension,
                  grid_depth='shallow'):
    """
    Creates a pandas data frame with labelled xyz coordinates according to
    the point spacing, and cubic dimension
    If the grid_depth == 'shallow', then there are four empty columns:
        'ASO', 'POS', 'NEG', 'POL'
    If the grid_depth == 'deep', then there are additional channels following those. In **order**
    {H: positive, sp2C: positive, sp3C: positive, sp2N: negative, sp3N: negative, sp2O: negative, sp3O: negative}
    For 11 total populate-able columns in 'deep' format

    Parameters
    ----------
    spacing : float
        Distance in Angstroms between grid points
    dimension : float
        Total size of grid: will be from -(dimension) to +dimension in x,y,z
    grid_depth = string
        Either 'shallow' or 'deep', will determine what kind of grid is generated

    Returns
    -------
    A pandas DataFrame that has evenly spaced x,y,z coordinates and the correct
    number/type of columns filled with zeros.

    """    
    
    x_val = np.arange((-1 * dimension), (dimension + 0.1), spacing)
    y_val = np.arange((-1 * dimension), (dimension + 0.1), spacing)
    z_val = np.arange((-1 * dimension), (dimension + 0.1), spacing)
    
    x_fin, y_fin, z_fin = np.meshgrid(x_val, y_val, z_val)
    
    x_fin = x_fin.ravel()
    y_fin = y_fin.ravel()
    z_fin = z_fin.ravel()
    
    fill_col = np.zeros(len(z_fin))
    
    if grid_depth == 'shallow':
        final_matrix = np.concatenate((x_fin,
                                      y_fin,
                                      z_fin,
                                      fill_col,
                                      fill_col,
                                      fill_col,
                                      fill_col),
                                      axis=0)
        final_matrix = final_matrix.reshape(7, -1).T
        empty_frame = pd.DataFrame(final_matrix,
                                   columns=['x',
                                            'y',
                                            'z',
                                            'ASO',
                                            'POL',
                                            'NEG',
                                            'POS'])
        return empty_frame
    
    if grid_depth == 'deep':
        final_matrix = np.concatenate((x_fin,
                                       y_fin,
                                       z_fin,
                                       fill_col,
                                       fill_col,
                                       fill_col,
                                       fill_col,
                                       fill_col, fill_col, fill_col, fill_col, fill_col, fill_col, fill_col),
                                       axis=0)
        final_matrix = final_matrix.reshape(14, -1).T
        empty_frame = pd.DataFrame(final_matrix,
                                   columns=['x',
                                            'y',
                                            'z',
                                            'ASO',
                                            'POL',
                                            'NEG',
                                            'POS',
                                            'H_POS', 'C_2_POS', 'C_3_POS', 'N_2_NEG', 'N_3_NEG', 'O_2_NEG', 'O_3_NEG'])
        return empty_frame
    
    
def aso_function(position1,
                 position2,
                 threshhold):
    """
    Calculate the euclidean distance between two xyz positions
    and set the 'steric occupany'
    
    
    Parameters
    ----------
    position1 : list or array
        list with [x, y, z] positions for atom at position 1
    position2 : list or array
        list with [x, y, z] positions for grid point at position 2
    threshhold: float
        van der Waals' radius for the atom of interest
    
    Returns
    -------
    Returns one if grid point 2 is inside VdW threshhold of atom 1,
    returns zero otherwise
    """
    
    distance = np.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2 + (position1[2] - position2[2])**2)
    
    if distance < threshhold:
        return 1
    else:
        return 0

def electro_function(position1, position2, charge):
    """
    Calculates the electrostatic potential at position1 with origin at position 2
    If inside VdW radius, entry will be set to zero during grid filling
    No set threshold - set during grid filling
    Returns the given positive or negative value
    Turned into absolute value when populating net charge grid points
    
    Parameters
    ----------
    position1 : list or array
        list with [x, y, z] positions for atom at position 1
    position2 : list or array
        list with [x, y, z] positions for grid point at position 2
    charge: float
        partial charge for the atom of interest
    
    Returns
    -------
    Returns calculated charge at position2 given atomic charge at position 1
    """
    
    distance = np.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2 + (position1[2] - position2[2])**2)
    return (charge / distance)

def polarize_function(position1, position2, polarizability):
    """
    Calculates the effective potential for polarizability at position1 with atom at position 2 (exponent 6!)
    If inside VdW radium, update will be set to zero by FillGrid function
    No threshold function: radius of update set in FillGrid
        
    Parameters
    ----------
    position1 : list or array
        list with [x, y, z] positions for atom at position 1
    position2 : list or array
        list with [x, y, z] positions for grid point at position 2
    polarizability: float
        static polarizability value for atom from Molecular Physics 2018 reference
        
    Returns
    -------
    Returns polarizability effect at position2 given atomic charge at position 1
    
    """
    
    distance = np.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2 + (position1[2] - position2[2])**2)
    return (polarizability / (distance**6))
    
def fill_shallow_grid(grid_size,
                      grid_spacing,
                      molecule,
                      conformer_list,
                      update_dist=7.5):
    """
    For a given grid size/spacing, create an empty frame
    For a given conformer list, update ASO, POL, POS, NEG
    Return the populated pandas DataFrame

    Parameters
    ----------
    grid_size : float
        Total size of grid in Å: will be from -(dimension) to +dimension in x,y,z
    grid_spacing : float
        Distance between grid points in Å.
    molecule : RDKit Molecule
        RDKit molecule object of calixarene being populated onto grid
    conformer_list : list
        A list of lists, where each entry is [population fraction, RDKit Conformer]
    update_dist : float, optional
        Cut-off distance for evaluating grid points. The default is 7.5.

    Returns
    -------
    A filled grid that represented all 'winning' conformers, weighed properly,
    for a given calixarene molecule.

    """    
    
    prop_list = []
    
    #All conformers are same order of atoms, so property list can be generated
    #from the first conformer
    
    #Populate properties in order of (vdw radius, charge, polarizability)
    
    for atom_object in molecule.GetAtoms():
        if atom_object.GetAtomicNum() == 1:
            prop_list.append((1.2, float(atom_object.GetProp('PartialCharge')), 4.5))
        elif atom_object.GetAtomicNum() == 6:
            prop_list.append((1.7, float(atom_object.GetProp('PartialCharge')), 11.3))
        elif atom_object.GetAtomicNum() == 7:
            prop_list.append((1.55, float(atom_object.GetProp('PartialCharge')), 7.4))
        elif atom_object.GetAtomicNum() == 8:
            prop_list.append((1.52, float(atom_object.GetProp('PartialCharge')), 5.3))
        elif atom_object.GetAtomicNum() == 9:
            prop_list.append((1.47, float(atom_object.GetProp('PartialCharge')), 3.7))
        elif atom_object.GetAtomicNum() == 16:
            prop_list.append((1.80, float(atom_object.GetProp('PartialCharge')), 19.4))
    
            
    empty_frame = generate_grid(grid_spacing, grid_size, 'shallow')
    
    for pop_conf in conformer_list:
        update_grid = generate_grid(grid_spacing, grid_size, 'shallow')
        work_atoms = pop_conf[1].GetPositions()
        for idx, atom in enumerate(work_atoms):
            #Update ASO Column
            aso_update_matrix = ((np.sqrt(((empty_frame['x'] - atom[0])**2) + ((empty_frame['y'] - atom[1])**2) + 
                                          ((empty_frame['z'] - atom[2])**2))) < prop_list[idx][0])
            #Conformer update grid used as filter for further populations
            update_grid.loc[aso_update_matrix, 'ASO'] = 1

        for idx, atom in enumerate(work_atoms):
            #Update POL column in empty frame (doesn't need an intermediate update grid)
            pol_update_matrix = ((update_grid['ASO'] == 0) & ((np.sqrt(((update_grid['x'] - atom[0])**2) + ((update_grid['y'] - atom[1])**2) + 
                                          ((update_grid['z'] - atom[2])**2))) < update_dist))
            empty_frame.loc[pol_update_matrix, 'POL'] += (pop_conf[0] * polarize_function((empty_frame.loc[pol_update_matrix, 'x'], empty_frame.loc[pol_update_matrix, 'y'],
                           empty_frame.loc[pol_update_matrix, 'z']), atom, prop_list[idx][2]))
            
            #Populate intermediate grid with net charge - use 'POS' column as a placeholder
            update_grid.loc[pol_update_matrix, 'POS'] += electro_function((empty_frame.loc[pol_update_matrix, 'x'], empty_frame.loc[pol_update_matrix, 'y'],
                           empty_frame.loc[pol_update_matrix, 'z']), atom, prop_list[idx][1])
        
        #At end of all atoms, update ASO, POS, NEG columns for conformer by inverse weight
        aso_equal_one = update_grid['ASO'] == 1
        empty_frame.loc[aso_equal_one, 'ASO'] += pop_conf[0]
        
        populate_pos = update_grid['POS'] > 0
        empty_frame.loc[populate_pos, 'POS'] += (update_grid.loc[populate_pos, 'POS'] * pop_conf[0])
        
        populate_neg = update_grid['POS'] < 0
        empty_frame.loc[populate_neg, 'NEG'] += abs(update_grid.loc[populate_neg, 'POS'] * pop_conf[0])
                    
    return empty_frame
            
def fill_deep_grid(grid_size,
                   grid_spacing,
                   molecule,
                   conformer_list,
                   update_dist=7.5):
    
    """            
    For a given grid size/spacing, create an empty frame
    For a given conformer list, update ASO, POL, POS, NEG and specific atom types
    Return the populated pandas DataFrame

    Parameters
    ----------
    grid_size : float
        Total size of grid in Å: will be from -(dimension) to +dimension in x,y,z
    grid_spacing : float
        Distance between grid points in Å.
    molecule : RDKit Molecule
        RDKit molecule object of calixarene being populated onto grid
    conformer_list : list
        A list of lists, where each entry is [population fraction, RDKit Conformer]
    update_dist : float, optional
        Cut-off distance for evaluating grid points. The default is 7.5.

    Returns
    -------
    A filled grid that represented all 'winning' conformers, weighed properly,
    for a given calixarene molecule.
    """    
    
    prop_list = []
    
    #All conformers are same order of atoms, so property list can be generated
    #from the first conformer
    
    #Populate properties in order of (vdw radius, charge, polarizability)
    
    for atom_object in molecule.GetAtoms():
        if atom_object.GetAtomicNum() == 1:
            prop_list.append((1.2, float(atom_object.GetProp('PartialCharge')), 4.5))
        elif atom_object.GetAtomicNum() == 6:
            if atom_object.GetHybridization() == Chem.HybridizationType.SP2:
                prop_list.append((1.7, float(atom_object.GetProp('PartialCharge')), 11.3, 'SP2'))
            elif atom_object.GetHybridization() == Chem.HybridizationType.SP3:
                prop_list.append((1.7, float(atom_object.GetProp('PartialCharge')), 11.3, 'SP3'))
            else:
                raise ValueError('Carbon atom does not have recognized hybridization')
        elif atom_object.GetAtomicNum() == 7:
            if atom_object.GetHybridization() == Chem.HybridizationType.SP2:
                prop_list.append((1.55, float(atom_object.GetProp('PartialCharge')), 7.4, 'SP2'))
            elif atom_object.GetHybridization() == Chem.HybridizationType.SP3:
                prop_list.append((1.55, float(atom_object.GetProp('PartialCharge')), 7.4, 'SP3'))
            else:
                raise ValueError('Nitrogen atom does not have recognized hybridization')
        elif atom_object.GetAtomicNum() == 8:
            if atom_object.GetHybridization() == Chem.HybridizationType.SP2:
                prop_list.append((1.52, float(atom_object.GetProp('PartialCharge')), 5.3, 'SP2'))
            elif atom_object.GetHybridization() == Chem.HybridizationType.SP3:
                prop_list.append((1.52, float(atom_object.GetProp('PartialCharge')), 5.3, 'SP3'))
            else:
                raise ValueError('Oxygen atom does not have recognized hybridization:', atom_object.GetHybridization())
        elif atom_object.GetAtomicNum() == 9:
            prop_list.append((1.47, float(atom_object.GetProp('PartialCharge')), 3.7))
        elif atom_object.GetAtomicNum() == 16:
            prop_list.append((1.80, float(atom_object.GetProp('PartialCharge')), 19.4))
    
            
    empty_frame = generate_grid(grid_spacing, grid_size, 'deep')
    
    for pop_conf in conformer_list:
        update_grid = generate_grid(grid_spacing, grid_size, 'deep')
        work_atoms = pop_conf[1].GetPositions()
        for idx, atom in enumerate(work_atoms):
            #Update ASO Column
            aso_update_matrix = ((np.sqrt(((empty_frame['x'] - atom[0])**2) + ((empty_frame['y'] - atom[1])**2) + 
                                          ((empty_frame['z'] - atom[2])**2))) < prop_list[idx][0])
            #Conformer update grid used as filter for further populations
            update_grid.loc[aso_update_matrix, 'ASO'] = 1

        for idx, atom in enumerate(work_atoms):
            #Update POL column in empty frame (doesn't need an intermediate update grid)
            pol_update_matrix = ((update_grid['ASO'] == 0) & ((np.sqrt(((update_grid['x'] - atom[0])**2) + ((update_grid['y'] - atom[1])**2) + 
                                          ((update_grid['z'] - atom[2])**2))) < update_dist))
            empty_frame.loc[pol_update_matrix, 'POL'] += (pop_conf[0] * polarize_function((empty_frame.loc[pol_update_matrix, 'x'], empty_frame.loc[pol_update_matrix, 'y'],
                           empty_frame.loc[pol_update_matrix, 'z']), atom, prop_list[idx][2]))
            
            #Populate intermediate grid with net charge - use 'POS' column as a placeholder
            update_grid.loc[pol_update_matrix, 'POS'] += electro_function((empty_frame.loc[pol_update_matrix, 'x'], empty_frame.loc[pol_update_matrix, 'y'],
                           empty_frame.loc[pol_update_matrix, 'z']), atom, prop_list[idx][1])

            if prop_list[idx][0] == 1.2 and prop_list[idx][1] > 0:
                update_grid.loc[pol_update_matrix, 'H_POS'] += electro_function((empty_frame.loc[pol_update_matrix, 'x'], empty_frame.loc[pol_update_matrix, 'y'],
                           empty_frame.loc[pol_update_matrix, 'z']), atom, prop_list[idx][1])

            elif prop_list[idx][0] == 1.7 and prop_list[idx][1] > 0:
                if prop_list[idx][3] == 'SP2':
                    update_grid.loc[pol_update_matrix, 'C_2_POS'] += electro_function((empty_frame.loc[pol_update_matrix, 'x'], empty_frame.loc[pol_update_matrix, 'y'],
                               empty_frame.loc[pol_update_matrix, 'z']), atom, prop_list[idx][1])
                elif prop_list[idx][3] == 'SP3':
                    update_grid.loc[pol_update_matrix, 'C_3_POS'] += electro_function((empty_frame.loc[pol_update_matrix, 'x'], empty_frame.loc[pol_update_matrix, 'y'],
                               empty_frame.loc[pol_update_matrix, 'z']), atom, prop_list[idx][1])
        
            elif prop_list[idx][0] == 1.55 and prop_list[idx][1] < 0:
                if prop_list[idx][3] == 'SP2':
                    update_grid.loc[pol_update_matrix, 'N_2_NEG'] += abs(electro_function((empty_frame.loc[pol_update_matrix, 'x'], empty_frame.loc[pol_update_matrix, 'y'],
                               empty_frame.loc[pol_update_matrix, 'z']), atom, prop_list[idx][1]))
                elif prop_list[idx][3] == 'SP3':
                    update_grid.loc[pol_update_matrix, 'N_3_NEG'] += abs(electro_function((empty_frame.loc[pol_update_matrix, 'x'], empty_frame.loc[pol_update_matrix, 'y'],
                               empty_frame.loc[pol_update_matrix, 'z']), atom, prop_list[idx][1]))
                    
            elif prop_list[idx][0] == 1.52 and prop_list[idx][1] < 0:
                if prop_list[idx][3] == 'SP2':
                    update_grid.loc[pol_update_matrix, 'O_2_NEG'] += abs(electro_function((empty_frame.loc[pol_update_matrix, 'x'], empty_frame.loc[pol_update_matrix, 'y'],
                               empty_frame.loc[pol_update_matrix, 'z']), atom, prop_list[idx][1]))
                elif prop_list[idx][3] == 'SP3':
                    update_grid.loc[pol_update_matrix, 'O_3_NEG'] += abs(electro_function((empty_frame.loc[pol_update_matrix, 'x'], empty_frame.loc[pol_update_matrix, 'y'],
                               empty_frame.loc[pol_update_matrix, 'z']), atom, prop_list[idx][1]))

        #At end of all atoms, update ASO, POS, NEG columns for conformer by inverse weight
        aso_equal_one = update_grid['ASO'] == 1
        empty_frame.loc[aso_equal_one, 'ASO'] += pop_conf[0]
        
        populate_pos = update_grid['POS'] > 0
        empty_frame.loc[populate_pos, 'POS'] += (update_grid.loc[populate_pos, 'POS'] * pop_conf[0])
        
        populate_neg = update_grid['POS'] < 0
        empty_frame.loc[populate_neg, 'NEG'] += abs(update_grid.loc[populate_neg, 'POS'] * pop_conf[0])
        
        #Fill in specific atom type columns
        pop_1h = update_grid['H_POS'] != 0
        empty_frame.loc[pop_1h, 'H_POS'] += (update_grid.loc[pop_1h, 'H_POS'] * pop_conf[0])
        
        pop_2c = update_grid['C_2_POS'] != 0
        empty_frame.loc[pop_2c, 'C_2_POS'] += (update_grid.loc[pop_2c, 'C_2_POS'] * pop_conf[0])
        
        pop_3c = update_grid['C_3_POS'] != 0
        empty_frame.loc[pop_3c, 'C_3_POS'] += (update_grid.loc[pop_3c, 'C_3_POS'] * pop_conf[0])
        
        pop_2n = update_grid['N_2_NEG'] != 0
        empty_frame.loc[pop_2n, 'N_2_NEG'] += (update_grid.loc[pop_2n, 'N_2_NEG'] * pop_conf[0])

        pop_3n = update_grid['N_3_NEG'] != 0
        empty_frame.loc[pop_3n, 'N_3_NEG'] += (update_grid.loc[pop_3n, 'N_3_NEG'] * pop_conf[0])

        pop_2o = update_grid['O_2_NEG'] != 0
        empty_frame.loc[pop_2o, 'O_2_NEG'] += (update_grid.loc[pop_2o, 'O_2_NEG'] * pop_conf[0])

        pop_3o = update_grid['O_3_NEG'] != 0
        empty_frame.loc[pop_3o, 'O_3_NEG'] += (update_grid.loc[pop_3o, 'O_3_NEG'] * pop_conf[0])
            
    return empty_frame
    
    
    
    
    