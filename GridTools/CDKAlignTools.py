#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 08:51:54 2021

@author: jvh
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import TransformConformer

def detect_isotope_nitrogen(molecule):
    """
    A function that detects the two 15N labelled nitrogen atoms in the
    constructed heterocycle. Further, the labelled nitrogen are identified
    as being with or without the key hydrogen atom attached.

    Parameters
    ----------
    molecule : RDKit Molecule object
        The actual heterocycle being analyzed

    Returns
    -------
    Two atom index numbers, labelled appropriately

    """
    nh_label = None
    n_ring_label = None
    
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 7:
            if atom.GetIsotope() == 15:
                nh_flag = False
                for curr_bond in atom.GetBonds():
                    if curr_bond.GetEndAtom().GetAtomicNum() == 1 or curr_bond.GetBeginAtom().GetAtomicNum() == 1:
                        nh_label = atom.GetIdx()
                        nh_flag = True
                if nh_flag == False:
                    n_ring_label = atom.GetIdx()
    
    if nh_label == None or n_ring_label == None:
        raise ValueError('Nitrogen isotope labels incorrectly assigned.\n',
                         'NH:', nh_label, 'and N_RING:', n_ring_label)
    
    return nh_label, n_ring_label

def key_carbon_idx(molecule,
                   nh_label,
                   n_ring_label):
    """
    A function that takes the two labelled nitrogen atoms and searches
    for the one carbon atom that is bonded to both. Returns the index
    of the key carbon atom

    Parameters
    ----------
    molecule : RDKit Molecule object
        Specific heterocycle being analyzed
    nh_label : integer
        Atom index of the 15N-H in heterocycle
    n_ring_label : integer
        Atom index of the 15N atom in heterocycle ring

    Returns
    -------
    Atom index of the carbon atom bonded to both 15NH and 15N ring

    """
    nh_atom = molecule.GetAtoms()[nh_label]
    n_ring_atom = molecule.GetAtoms()[n_ring_label]
    
    nh_bonds = nh_atom.GetBonds()
    n_ring_bonds = n_ring_atom.GetBonds()
    
    nh_other = []
    n_ring_other = []
    
    shared_atoms = []
    
    for bond in nh_bonds:
        if bond.GetBeginAtomIdx() not in nh_other and bond.GetBeginAtomIdx() != nh_atom:
            nh_other.append(bond.GetBeginAtomIdx())
        if bond.GetEndAtomIdx() not in nh_other and bond.GetEndAtomIdx() != nh_atom:
            nh_other.append(bond.GetEndAtomIdx())
    
    for bond in n_ring_bonds:
        if bond.GetBeginAtomIdx() not in n_ring_other and bond.GetBeginAtomIdx() != n_ring_atom:
            n_ring_other.append(bond.GetBeginAtomIdx())
        if bond.GetEndAtomIdx() not in n_ring_other and bond.GetEndAtomIdx() != n_ring_atom:
            n_ring_other.append(bond.GetEndAtomIdx())
    
    for atom_idx in nh_other:
        if atom_idx in n_ring_other:
            shared_atoms.append(atom_idx)
    
    if len(shared_atoms) == 0:
        raise ValueError('No shared atoms identified in heterocycle')
    elif len(shared_atoms) > 1:
        raise ValueError('More than one shared atom identified in heterocycle')
    else:
        return shared_atoms[0]
    
def gen_translate_array(conformer,
                        nh_label):
    """
    A function that takes a single RDKit conformer, and the atom IDX of the
    15N-H atom, and creates a translation transformation that will place the
    nitrogen atom at coordinates (0, 0, 0)

    Parameters
    ----------
    conformer : RDKit conformer object
        The specific heterocycle that will have NH translated to the origin
    nh_atom_idx : integer
        The atom IDX of the 15N labelled NH group

    Returns
    -------
    An array that can be used by RDKit to translate the conformer as desired.

    """    
            
    x_trans = -(conformer.GetAtomPosition(nh_label).x)
    y_trans = -(conformer.GetAtomPosition(nh_label).y)
    z_trans = -(conformer.GetAtomPosition(nh_label).z)

    translation_array = np.array([[1, 0, 0, x_trans],
                                  [0, 1, 0, y_trans],
                                  [0, 0, 1, z_trans],
                                  [0, 0, 0, 1]])
    
    return translation_array
    
def gen_zrot_array(conformer,
                   shared_atom_label):
    """
    A function that takes the shared carbon atom (assumes that the 15N-H group is
    already at the 0,0,0 origin) and rotates it along the z-axis, until the bond
    to the shared atom is lying along the y-axis. That is: the x-position of this
    shared atom is zero.

    Parameters
    ----------
    conformer : RDKit conformer object
        The specific heterocycle conformer being oriented
    shared_atom_label : integer
        The index for the unique atom shared between both 15N labelled atoms

    Returns
    -------
    An array that can be used by RDKit to rotate the molecule as desired

    """    
    
    x_pos = conformer.GetAtomPosition(shared_atom_label).x
    y_pos = conformer.GetAtomPosition(shared_atom_label).y
 
    #First, calculate polar coordinates

    azimuth = float(np.arctan(np.array([x_pos / y_pos])))
    
    zrot_array = np.array([[np.cos(azimuth), -np.sin(azimuth), 0, 0],
                           [np.sin(azimuth), np.cos(azimuth), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]) 
    
    return zrot_array
    
def gen_xrot_array(conformer,
                   shared_atom_label):
    """
    A function that takes a conformer already translated (NH to 0,0,0) and
    rotated around the x-axis (to shared atom.x = 0). Calculates the rotation
    around the x-axis that places the 15NH-shared_atom bond running along
    the z-axis (shared_atom.x and .y are both zero)

    Parameters
    ----------
    conformer : RDKit conformer object
        The specific heterocycle conformer being aligned
    shared_atom_label : integer
        The atom index of the unique atom shared between both 15N labelled atoms

    Returns
    -------
    An array that can be used by RDKit to rotate the molecule as desired

    """    
    
    y_pos = conformer.GetAtomPosition(shared_atom_label).y
    z_pos = conformer.GetAtomPosition(shared_atom_label).z

    rot_amt = float(np.arctan(np.array([y_pos / z_pos])))

    xrot_array = np.array([[1, 0, 0, 0],
                           [0, np.cos(rot_amt), -np.sin(rot_amt), 0],
                           [0, np.sin(rot_amt), np.cos(rot_amt), 0],
                           [0, 0, 0, 1]])
    
    return xrot_array

def z_direction_check(conformer,
                      shared_atom_label):
    """
    A function that takes a heterocycle with the bond between 15N-H and the
    unique shared atom running along the z-axis. Check to see if the z-position
    of the shared atom is positive or negative. If negative, rotate by 180 degrees
    around the y-axis

    Parameters
    ----------
    conformer : RDKit conformer object
        The specific heterocycle conformer being aligned
    shared_atom_label : integer
        The atom index of the unique atom shared between 15N labelled atoms

    Returns
    -------
    None - but RDKit conformer object is adjusted in place to desired orientation

    """

    z_value = conformer.GetAtomPosition(shared_atom_label).z
    
    if z_value < 0:
        
        rotation = np.radians(180)
        
        z_flip_array = np.array([[np.cos(rotation), 0, np.sin(rotation), 0],
                                 [0, 1, 0, 0],
                                 [-np.sin(rotation), 0, np.cos(rotation), 0],
                                 [0, 0, 0, 1]])
        
        TransformConformer(conformer, z_flip_array)
    
    return

    
def gen_ring_rotation_array(conformer,
                            n_ring_label):
    """
    Takes a conformer that has been rotate to have (0,0,0) for 15N-H atom,
    (0,0,non-zero) for the unique shared atom. Rotates along the z-axis so
    that the second 15N lies on the y-axis (x value of zero).

    Parameters
    ----------
    conformer : RDKit conformer object
        Specific heterocycle conformer being rotated
    n_ring_label : integer
        The atom label for the 15N labelled nitrogen atom in the heterocycle ring

    Returns
    -------
    An array that can be used by RDKit to rotate the molecule as desired.

    """    
    
    x_pos = conformer.GetAtomPosition(n_ring_label).x
    y_pos = conformer.GetAtomPosition(n_ring_label).y

    rot_amt = float(np.arctan(np.array([x_pos / y_pos])))
        
    ring_rot_array = np.array([[np.cos(rot_amt), -np.sin(rot_amt), 0, 0],
                               [np.sin(rot_amt), np.cos(rot_amt), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    return ring_rot_array
    
def ring_conf_flip(conformer,
                   n_ring_label,
                   flip_flag):
    """
    A function that checks to see if the 15N in the hereocycle ring has a positive
    or negative value on the y-axis. Default setting (flip_flag = False) wants a
    positive value for this number - rotate by 180 degrees if incorrect. If flip_flag
    is true and positive value on y-axis, then also rotate 180 degrees.

    Parameters
    ----------
    conformer : RDKit conformer object
        Specific heterocycle conformer being checked and perhaps rotated
    n_ring_label : integer
        Atom label of the 15N atom found in the heterocycle ring
    flip_flag : Boolean
        Flag set by conformer generating workflow. Eventually, both 'True' and 'False'
        are called, and the two different batched of conformers are saved as '_U' (for 'up',
        when the flag is False) or '_D' when the flag is True.

    Returns
    -------
    None - but the RDKit Conformer object is adjusted in place if necessary

    """    
    
    y_pos = conformer.GetAtomPosition(n_ring_label).y
    
    rotation = np.radians(180)
    
    flip_rotation = np.array([[np.cos(rotation), -np.sin(rotation), 0, 0],
                              [np.sin(rotation), np.cos(rotation), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
    
    if y_pos < 0:
        if flip_flag == False:
            TransformConformer(conformer, flip_rotation)
    else:
        if flip_flag == True:
            TransformConformer(conformer, flip_rotation)
    
    return

def align_single_conf(conformer,
                      molecule,
                      flip_flag):
    """
    Aligns a single conformer of a molecule, such that the 15N-H is at the origin,
    its bond to the unique shared atom runs along the z-axis

    Parameters
    ----------
    conformer : RDKit Conformer object
        Specific heterocycle conformer being aligned
    molecule : RDKit molecule object
        The parent molecule of the conformer being aligned
    flip_flag : Boolean
        Determines whether 'Up' or 'Down' conformer is generated

    Returns
    -------
    None - but RDKit conformer object is adjusted in place to desired orientation

    """
    
    nh_label, n_ring_label = detect_isotope_nitrogen(molecule)

    shared_atom_label = key_carbon_idx(molecule,
                                       nh_label,
                                       n_ring_label)

    translation_array = gen_translate_array(conformer,
                                            nh_label)
    TransformConformer(conformer, translation_array)

    zrot_array = gen_zrot_array(conformer,
                                shared_atom_label)
    TransformConformer(conformer, zrot_array)

    xrot_array = gen_xrot_array(conformer,
                                shared_atom_label)
    TransformConformer(conformer, xrot_array)

    z_direction_check(conformer,
                      shared_atom_label)

    ring_rot_array = gen_ring_rotation_array(conformer,
                                             n_ring_label)
    TransformConformer(conformer, ring_rot_array)

    ring_conf_flip(conformer,
                   n_ring_label,
                   flip_flag)

    return

def align_all_confs(molecule,
                    flip_flag):
    """
    Aligns all conformers in a molecule by simply making a list of conformers
    and performing align_sing_conf() iteratively

    Parameters
    ----------
    molecule : RDKit molecule
        An RDKit molecule to be aligned one conformer at a time
    flip_flag : Boolean
        Determines whether 'Up' or 'Down' conformer is generated

    Returns
    -------
    None - but all RDKit Conformer objects associated with the molecule are adjusted

    """    
    
    for single_conf in molecule.GetConformers():
        align_single_conf(single_conf,
                          molecule,
                          flip_flag)
    
    return

# ------------------------
#         Unit Test
# ------------------------
    
def quality_check_print(conformer,
                        nh_label,
                        n_ring_label,
                        shared_atom_label):
    """
    A print function that outputs the x,y,z coordinates of the key atoms in molecule

    Parameters
    ----------
    conformer : RDKit conformer object
        Actual conformer being adjusted
    nh_label : integer
        Atom index of the 15N-H nitrogen atom
    n_ring_label : integer
        Atom index of the 15N nitrogen in the heterocycle ring
    shared_atom_label : integer
        Atom index of the unique atom shared between 15N atoms

    Returns
    -------
    None - prints key atoms positions
    
    """    
    
    atom_dict = {'15N-H': nh_label,
                   '15N-Ring': n_ring_label,
                   'Shared Atom': shared_atom_label}
    
    for key_atom in atom_dict:
        x = conformer.GetAtomPosition(atom_dict[key_atom]).x
        y = conformer.GetAtomPosition(atom_dict[key_atom]).y
        z = conformer.GetAtomPosition(atom_dict[key_atom]).z

        print ('Key atom', str(key_atom), 'is at: ', x, " , ", y, " , ", z)
        
    return

def single_conf_check(conformer,
                      molecule,
                      flip_flag):
    """
    Performs the exact same alignment process as above, but prints the
    atoms positions at every step to ensure the math was done correctly/
    alignment is appropriate. 

    Parameters
    ----------
    conformer : RDKit Conformer object
        The specific conformer to be aligned
    molecule : RDKit Molecule object
        The parent molecule of the specific conformer to be aligned
    flip_flag : Boolean
        Determines whether 'Up' or 'Down' conformer is generated

    Returns
    -------
    None - but prints key positions at every step of process

    """    
    
    nh_label, n_ring_label = detect_isotope_nitrogen(molecule)

    shared_atom_label = key_carbon_idx(molecule,
                                       nh_label,
                                       n_ring_label)

    print('Starting atom positions, with flip_flag = ', flip_flag)

    quality_check_print(conformer,
                        nh_label,
                        n_ring_label,
                        shared_atom_label)
    
    translation_array = gen_translate_array(conformer,
                                            nh_label)
    TransformConformer(conformer, translation_array)
    
    print('.')
    print('After translation')
    
    quality_check_print(conformer,
                        nh_label,
                        n_ring_label,
                        shared_atom_label)

    zrot_array = gen_zrot_array(conformer,
                                shared_atom_label)
    TransformConformer(conformer, zrot_array)

    print('.') 
    print('After z-rotation')

    quality_check_print(conformer,
                        nh_label,
                        n_ring_label,
                        shared_atom_label)

    xrot_array = gen_xrot_array(conformer,
                                shared_atom_label)
    TransformConformer(conformer, xrot_array)

    print('.')
    print('After x-rotation')

    quality_check_print(conformer,
                        nh_label,
                        n_ring_label,
                        shared_atom_label)

    z_direction_check(conformer,
                      shared_atom_label)

    print('.')
    print('After z-direction check')

    quality_check_print(conformer,
                        nh_label,
                        n_ring_label,
                        shared_atom_label)

    ring_rot_array = gen_ring_rotation_array(conformer,
                                             n_ring_label)
    TransformConformer(conformer, ring_rot_array)

    print('.')
    print('After ring rotation')

    quality_check_print(conformer,
                        nh_label,
                        n_ring_label,
                        shared_atom_label)
    
    ring_conf_flip(conformer,
                   n_ring_label,
                   flip_flag)

    print('.')
    print('After flip-flag check')

    quality_check_print(conformer,
                        nh_label,
                        n_ring_label,
                        shared_atom_label)

    return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    