#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 20:48:50 2020

@author: jvh

AlignTools are functions used to align the calixarene molecules for the Hof
lab collaboration project

"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import TransformConformer


def all_calix_phenol(molecule):
    """
    A function that detects all four calixarene phenols - made easy by the
    fact that the key oxygen atoms have been labelled as isotope 17-O

    Parameters
    ----------
    molecule : RDKit Molecule object
        The actual calixarene being analyzed

    Returns
    -------
    A list of atom IDX's for the four 17-O labelled oxygen atoms

    """

    phenol_list = []
    
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 8:
            if atom.GetIsotope() == 17:
                phenol_list.append(atom.GetIdx())
    return phenol_list

def iso_deprot_phenol(molecule,
                      phenol_list):
    """
    A method that looks at a list of (isotopically labelled) phenol atoms,
    and selects the atom IDX for the unique example with only one bond

    Parameters
    ----------
    molecule : RDKit Molecule object
        The actual calixarene being analyzed    
    phenol_list : list
        A list of oxygen atom IDX from 17-O isotope labels

    Returns
    -------
    An integer - that is the unique IDX of the deprotonated phenol and a list
    with the other IDX of the -OH phenols

    """
    deprot_list = []
    oh_list = []
    
    for idx in phenol_list:
        if len(molecule.GetAtomWithIdx(idx).GetBonds()) == 1:
            deprot_list.append(idx)
        else:
            oh_list.append(idx)
        
    if len(deprot_list) > 1:
        raise ValueError('More than one labelled oxygen is deprotonated')
    elif len(deprot_list) == 1:
        return deprot_list[0], oh_list
    else:
        raise ValueError('No labelled oxygen atoms are deprotonated')

    return            

def idx_13c(molecule):
    """
    Function that detects the unique 13C labelled carbon atom that is used for
    orienting around the z-axis

    Parameters
    ----------
    molecule : RDKit Molecule object
        The actual calixarene being analyzed

    Returns
    -------
    The atom idx of the 13C labelled carbon atom

    """

    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 6:
            if atom.GetIsotope() == 13:
                return atom.GetIdx()
    
    raise ValueError('No 13C labelled carbon atom detected')
    
def atom_dist(conformer,
              atom1,
              atom2):
    """
    A quick function that returns the Euclidean distance between two atoms,
    given the conformer

    Parameters
    ----------
    conformer : RDKit Conformer object
        The calixarene conformer that is being assessed
    atom1 : integer
        IDX of the first atom of the pair being measured
    atom2 : integer
        IDX of the second atom of the pair being measured

    Returns
    -------
    A float: the distance in angstroms between the atoms of interest
    
    """

    positions = conformer.GetPositions()
    pos1 = positions[atom1]
    pos2 = positions[atom2]
    
    distance = np.sqrt(((pos1[0] - pos2[0])**2) + ((pos1[1] - pos2[1])**2) + ((pos1[2] - pos2[2])**2))

    return distance

def define_measure_plane(conformer,
                         phenol_idx,
                         phenol_list):
    """
    From the atom IDX of a unique deprotonated phenol, and from a list of
    -OH phenols, construct a plane from the deprotonated phenol and the
    next two closest phenols. Measure the distance from the third -OH
    phenol to the plane. Use this distance as a quality control check.

    Parameters
    ----------
    conformer: RDKit Conformer object
        The actual calixarene conformer that is being measured in 3D
    phenol_idx : integer
        The atom IDX for the unique deprotonated phenol
    phenol_list : list
        A list of atom IDX for all -OH phenols in molecule

    Returns
    -------
    A value for the distance between the third -OH phenol and the other
    defined plane and the sorted list.

    """
    #Create list of tuples of form [(atom_idx, distance)], sorted by distance
    distance_list = []
    
    for phenol in phenol_list:
        this_dist = atom_dist(conformer, phenol_idx, phenol)
        distance_list.append((phenol, this_dist))
    
    distance_list.sort(key=lambda x: x[1])
    
    point_1 = (conformer.GetAtomPosition(phenol_idx).x,
               conformer.GetAtomPosition(phenol_idx).y,
               conformer.GetAtomPosition(phenol_idx).z)
    
    point_2 = (conformer.GetAtomPosition(distance_list[0][0]).x,
               conformer.GetAtomPosition(distance_list[0][0]).y,
               conformer.GetAtomPosition(distance_list[0][0]).z)
    
    point_3 = (conformer.GetAtomPosition(distance_list[1][0]).x,
               conformer.GetAtomPosition(distance_list[1][0]).y,
               conformer.GetAtomPosition(distance_list[1][0]).z)
    
    test_pt = (conformer.GetAtomPosition(distance_list[2][0]).x,
               conformer.GetAtomPosition(distance_list[2][0]).y,
               conformer.GetAtomPosition(distance_list[2][0]).z)
    
    a1 = (point_2[0] - point_1[0])
    b1 = (point_2[1] - point_1[1])
    c1 = (point_2[2] - point_1[2])
    a2 = (point_3[0] - point_1[0])
    b2 = (point_3[1] - point_1[1])
    c2 = (point_3[2] - point_1[2])
    
    a = (b1 * c2) - (b2 * c1)
    b = (a2 * c1) - (a1 * c2)
    c = (a1 * b2) - (b1 * a2)
    d = -1 * ((a * point_1[0]) - (b * point_1[1]) - (c * point_1[2]))
    
    numerator = abs((a * test_pt[0]) + (b * test_pt[1]) + (c * test_pt[2]) + d)
    denomer = np.sqrt((a**2) + (b**2) + (c**2))
    distance = (numerator / denomer)
    
    return distance, distance_list

            
def key_carbon_idx(molecule,
                   true_phenol_idx):
    """
    A function that takes an RDKit Molecule object, and given the already
    discovered atom IDX of the deprotonated phenol (from detect_deprot_phenol())
    returns a dictionary with four entries:
     {'phenol': IDX of deprotonated phenol
     'ipso': IDX of attached carbon atom
     'ring1': IDX for first ring carbon atom
     'ring2': IDX of second ring carbon atom}

    Parameters
    ----------
    molecule : RDKit Molecule object
        The specific calixarene being analyzed
    true_phenol_idx : integer
        Atom IDX of the unique calixarene deprotonated phenol

    Returns
    -------
    A dictionary of the format described above
    
    """
    
    phenol_atom = molecule.GetAtoms()[true_phenol_idx]
    
    phenol_bond = phenol_atom.GetBonds() 
    
    #Check to make sure deprotonated phenol was assigned correctly
    if len(phenol_bond) > 1:
        raise ValueError('Deprotonated phenol was assigned to atom with more than 1 bond')        
        quit
    
    #Detect and assign ring carbon
    if phenol_bond[0].GetBeginAtom().GetIsAromatic() == True:
        ipso_atom = phenol_bond[0].GetBeginAtomIdx()
    elif phenol_bond[0].GetEndAtom().GetIsAromatic() == True:
        ipso_atom = phenol_bond[0].GetEndAtomIdx()
    
    #Detect and assign ortho carbon atoms in ring
    ipso_bonds = molecule.GetAtoms()[ipso_atom].GetBonds()
    in_ring_idx = []
    
    for bond_obj in ipso_bonds:
        if bond_obj.GetBeginAtom().GetIsAromatic() == True and bond_obj.GetBeginAtom().GetAtomicNum() == 6:
            if bond_obj.GetBeginAtom().GetIdx() != ipso_atom:
                in_ring_idx.append(bond_obj.GetBeginAtom().GetIdx())
                    
        if bond_obj.GetEndAtom().GetIsAromatic() == True and bond_obj.GetEndAtom().GetAtomicNum() == 6:
            if bond_obj.GetEndAtom().GetIdx() != ipso_atom:
                in_ring_idx.append(bond_obj.GetEndAtom().GetIdx())
    
    #Check to make sure 2 and only 2 in_ring_idx are assigned
    if len(in_ring_idx) != 2:
        raise ValueError('Incorrect number (' + len(in_ring_idx) + ') of in_ring carbon atoms assigned')
        quit
    
    return_dict = {'phenol': true_phenol_idx,
                   'ipso': ipso_atom,
                   'ring1': in_ring_idx[0],
                   'ring2': in_ring_idx[1]}
    
    return return_dict
                    

def gen_translate_array(conformer,
                        phenol_idx):
    """
    A function that takes a single RDKit conformer, and the atom IDX of the
    unique deprotonated phenol, and creates a translation transformation that
    will place the deprotonated phenol at coordinates (0, 0, 0)

    Parameters
    ----------
    conformer : RDKit Conformer object
        The specific calixarene conformer that will be oriented onto the origin
    phenol_idx : integer
        The atom IDX of the unique deprotonated phenol.

    Returns
    -------
    An array that can be used by RDKit to translate the molecule as desired

    """    

    x_trans = -(conformer.GetAtomPosition(phenol_idx).x)
    y_trans = -(conformer.GetAtomPosition(phenol_idx).y)
    z_trans = -(conformer.GetAtomPosition(phenol_idx).z)

    translation_array = np.array([[1, 0, 0, x_trans],
                                  [0, 1, 0, y_trans],
                                  [0, 0, 1, z_trans],
                                  [0, 0, 0, 1]])
    
    return translation_array
            
def gen_zrot_array(conformer,
                   ipso_atom_idx):
    """
    A function that takes the ipso carbon atom (assumes that the deprotonated phenol
    is already at (0, 0, 0) location) and rotates it around the z-axis, until the C-O
    bond is lying along the y-axis: that is, the 'x' position of this carbon is zero.

    Parameters
    ----------
    conformer : RDKit Conformer object
        The specific calixarene conformer that will be oriented onto the origin
    ipso_atom_idx : integer
        The atom IDX of the carbon directly connected to the unique deprotonated phenol

    Returns
    -------
    An array that can be used by RDKit to rotate the molecule as desired.

    """
    
    x_pos = conformer.GetAtomPosition(ipso_atom_idx).x
    y_pos = conformer.GetAtomPosition(ipso_atom_idx).y
 
    #First, calculate polar coordinates

    azimuth = float(np.arctan(np.array([x_pos / y_pos])))
    
    zrot_array = np.array([[np.cos(azimuth), -np.sin(azimuth), 0, 0],
                           [np.sin(azimuth), np.cos(azimuth), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]) 
    
    return zrot_array

def gen_xrot_array(conformer,
                   ipso_atom_idx):
    """
    Take molecule that has already been translated and rotate around z axis.
    Calculate the rotation around the X axis necessary to place the C-O bond
    running along z-axis (x and y value of C and O is ~zero)

    Parameters
    ----------
    conformer : RDKit Conformer object
        The specific calixarene conformer that will be oriented onto the origin
    ipso_atom_idx : integer
        The atom IDX of the carbon directly connected to the unique deprotonated phenol

    Returns
    -------
    An array that can be used by RDKit to rotate the molecule as desired.

    """
    
    y_pos = conformer.GetAtomPosition(ipso_atom_idx).y
    z_pos = conformer.GetAtomPosition(ipso_atom_idx).z

    rot_amt = float(np.arctan(np.array([y_pos / z_pos])))

    xrot_array = np.array([[1, 0, 0, 0],
                           [0, np.cos(rot_amt), -np.sin(rot_amt), 0],
                           [0, np.sin(rot_amt), np.cos(rot_amt), 0],
                           [0, 0, 0, 1]])
    
    return xrot_array

def gen_ring_rotation_array(conformer,
                            ring_atom1_idx,
                            ring_atom2_idx):
    """
    Takes a molecule that has been rotated to have [0, 0, 0] for phenol,
    [0, 0, non-zero] for ring junction carbon atoms.
    Rotates along Z axis so that aromatic ring lies along the y-axis
    Do not use average value! Sometimes, when closely aligned in above steps,
    you get rotations of 179, and 1 degree, which then averages to 90.

    Parameters
    ----------
    conformer : RDKit Conformer Object 
        The specific calixarene conformer to be rotated.
    ring_atom1_idx : integer
        The atom IDX for the first of two ortho-carbon atoms
    ring_atom2_idx : integer
        The atom IDX for the second of two ortho-carbon atoms

    Returns
    -------
    An array that can be used by RDKit to rotate the molecule as desired.

    """

    x_pos_1 = conformer.GetAtomPosition(ring_atom1_idx).x
    y_pos_1 = conformer.GetAtomPosition(ring_atom1_idx).y
        
    rot_amt_1 = float(np.arctan(np.array([x_pos_1 / y_pos_1])))
    
    to_rotate = rot_amt_1 
    
    ring_rot_array = np.array([[np.cos(to_rotate), -np.sin(to_rotate), 0, 0],
                               [np.sin(to_rotate), np.cos(to_rotate), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    return ring_rot_array

def z_axis_check(conformer,
                 ipso_atom_idx):
    """
    Takes a conformer that has been fully rotated and aligned
    Checks second atom idx (carbon atom connected to phenolic oxygen)
    Flips molecules if that atom is at a negative position

    Parameters
    ----------
    conformer : RDKit Conformer object
        The specific calixarene conformer that will be oriented onto the origin
    ipso_atom_idx : integer
        The atom IDX of the carbon directly connected to the unique deprotonated phenol

    Returns
    -------
    None - but RDKit Conformer object is adjusted in place if necessary

    """
    
    z_pos = conformer.GetAtomPosition(ipso_atom_idx).z    
    
    if z_pos < 0:
        rotation = np.radians(180)
        
        flip_rotation = np.array([[np.cos(rotation), 0, np.sin(rotation), 0],
                                  [0, 1, 0, 0],
                                  [-np.sin(rotation), 0, np.cos(rotation), 0],
                                  [0, 0, 0, 1]])
        
        TransformConformer(conformer, flip_rotation)
    
    return

def x_axis_check(conformer,
                 molecule):
    """
    Takes a molecule that has been fully rotated and aligned, z-axis checked
    If the 13C labelled carbon is negative, rotate by 180 degrees around z axis    

    Parameters
    ----------
    conformer : RDKit Conformer object
        The specific calixarene conformer that will be oriented onto the origin
    molecule : RDKit Molecule object
        Conformer objects do not stores lists of atoms: parent molecule must
        be accessed to give full atom list

    Returns
    -------
    None - but RDKit Conformer is adusted in place if necessary

    """
    
    labelled_13c = idx_13c(molecule)
    
    x_value = conformer.GetAtomPosition(labelled_13c).x

    if x_value < 0:
        rotation = np.radians(180)
        
        flip_rotation = np.array([[np.cos(rotation), -np.sin(rotation), 0, 0],
                                  [np.sin(rotation), np.cos(rotation), 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
        TransformConformer(conformer, flip_rotation)
        print('X-axis rotated')
    return

def align_single_conf(conformer,
                      molecule):
    """
    Aligns a single conformer of a molecule, such that the C-O bond of the
    unique deprotonated phenol runs in the positive z-direction
    The aromatic ring of that phenol lies along the y-axis
    The bulk of the molecule lies on the positive side of the y-axis

    Parameters
    ----------
    conformer : RDKit Conformer object
        The specific conformer to be aligned
    molecule : RDKit Molecule object
        The parent molecule of the specific conformer ot be aligned

    Returns
    -------
    None - but RDKit Conformer object is adjusted in place to desired orientation

    """        

    phenol_list = all_calix_phenol(molecule)
    true_phenol_idx, oh_phenol_list = iso_deprot_phenol(molecule,
                                                        phenol_list)
    
    key_atom_dict = key_carbon_idx(molecule,
                                   true_phenol_idx)
    
    
    translation_array = gen_translate_array(conformer,
                                            key_atom_dict['phenol'])
    TransformConformer(conformer, translation_array)
    
    
    zrot_array = gen_zrot_array(conformer,
                                key_atom_dict['ipso'])
    TransformConformer(conformer, zrot_array)
    
    
    xrot_array = gen_xrot_array(conformer,
                                key_atom_dict['ipso'])
    TransformConformer(conformer, xrot_array)
    
    ring_rot_array = gen_ring_rotation_array(conformer,
                                             key_atom_dict['ring1'],
                                             key_atom_dict['ring2'])
    TransformConformer(conformer, ring_rot_array)
    
    
    z_axis_check(conformer,
                 key_atom_dict['ipso'])
    
    
    x_axis_check(conformer,
                 molecule)
    
    
    return    

def align_all_confs(molecule):
    """
    Aligns all conformers in a molecule, by simply making a list of conformers
    and applying align_single_conf() iteratively

    Parameters
    ----------
    molecule : RDKit Molecule
        An RDKit molecule to be aligned one conformer at a time

    Returns
    -------
    None - but all RDKit Conformer objects associated with the molecule are adjusted

    """

    for single_conf in molecule.GetConformers():
        align_single_conf(single_conf,
                          molecule)
    
    return

#-------------------
#    Unit Tests
#-------------------

def quality_check_print(conformer,
                        atom_dict):
    """
    A quick print function that gives the location of the four key atoms during
    alignment adjustments.

    Parameters
    ----------
    conformer : RDKit Conformer object 
        Actual calixarene confromer that is being aligned.
    atom_dict : dictionary
        Dictionary with four atom IDXs, 'phenol', 'ipso', 'ring1', and 'ring2'

    Returns
    -------
    None - prints key atom positions

    """

    for key_atom in atom_dict:
        x = conformer.GetAtomPosition(atom_dict[key_atom]).x
        y = conformer.GetAtomPosition(atom_dict[key_atom]).y
        z = conformer.GetAtomPosition(atom_dict[key_atom]).z

        print ('Key atom', str(key_atom), 'is at: ', x, " , ", y, " , ", z)
        
    return


def single_conf_check(conformer,
                      molecule):
    """
    Performs the exact same alignment as the above scripts - but prints
    the positions of the key atoms at every step to ensure that
    math was done correctly/alignment is appropriate

    Parameters
    ----------
    conformer : RDKit Conformer object
        The specific conformer to be aligned
    molecule : RDKit Molecule object
        The parent molecule of the specific conformer ot be aligned

    Returns
    -------
    None - but RDKit Conformer object is adjusted in place to desired orientation
    and positions of key atoms are printed at each stage of alignment

    """


    phenol_list = all_calix_phenol(molecule)
    true_phenol_idx, oh_phenol_list = iso_deprot_phenol(molecule,
                                                        phenol_list)
    
    key_atom_dict = key_carbon_idx(molecule,
                                   true_phenol_idx)
    
    print('Starting Atom Positions')
    quality_check_print(conformer,
                        key_atom_dict)
    
    translation_array = gen_translate_array(conformer,
                                            key_atom_dict['phenol'])
    TransformConformer(conformer, translation_array)
    
    print('After translation')
    quality_check_print(conformer,
                        key_atom_dict)
    
    
    zrot_array = gen_zrot_array(conformer,
                                key_atom_dict['ipso'])
    TransformConformer(conformer, zrot_array)
    
    print('After z-rotation')
    quality_check_print(conformer,
                        key_atom_dict)
    
    
    xrot_array = gen_xrot_array(conformer,
                                key_atom_dict['ipso'])
    TransformConformer(conformer, xrot_array)
    
    print('After x-rotation')
    quality_check_print(conformer,
                        key_atom_dict)
    
    
    ring_rot_array = gen_ring_rotation_array(conformer,
                                             key_atom_dict['ring1'],
                                             key_atom_dict['ring2'])
    TransformConformer(conformer, ring_rot_array)
    print('After ring rotation')
    quality_check_print(conformer,
                        key_atom_dict)
    
    
    
    z_axis_check(conformer,
                 key_atom_dict['ipso'])
    print('After z-axis check')
    quality_check_print(conformer,
                        key_atom_dict)
    
    
    x_axis_check(conformer,
                 molecule)
    print('After y-axis check')
    quality_check_print(conformer,
                        key_atom_dict)
    
    
    return    


