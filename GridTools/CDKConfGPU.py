#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:13:09 2020

@author: jvh

New file for conformer generation, minimization, ranking
    atom_tens = torch.LongTensor(atom_list).cuda()
    atom_tens.unsqueeze_(0)
    .squeeze().detach().numpy()
    calc = torchani.models.ANI2x(periodic_table_index=True).ase().to('cuda')


"""

import torch
import torchani
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Geometry import Point3D
import ase
from ase.optimize import BFGS
from ase.optimize import LBFGS
import sys, os
import time

def load_ani2():
    """
    Self-explanatory

    Returns
    -------
    Loads the ANI2x model and returns the pre-trained model object

    """

    model = torchani.models.ANI2x(periodic_table_index=True).to('cuda')
    
    return model

def conf_to_torch(molecule,
                  conformer):
    """
    A function that takes a single molecule conformer, and converts it into
    the correct format of a Pytorch tensor for energy calculation/minimization

    Parameters
    ----------
    molecule: RDKit Molecule object
        The parent molecule fo the conformer being assessed
    conformer : RDKit Conformer object
        The single conformer that is having its energy assessed

    Returns
    -------
    Two Pytorch tensors, one of the correct structure with atom positions,
    one list of atom types: in some cases this will be converted into
    

    """    
    
    pos_array = conformer.GetPositions()
    pos_tens = torch.FloatTensor(pos_array).to('cuda')
    #Add batch size of one in place
    pos_tens.unsqueeze_(0)
    pos_tens.requires_grad = True
    
    atom_list = []
    for atom in molecule.GetAtoms():
        atom_list.append(atom.GetAtomicNum())
    
    atom_tens = torch.LongTensor(atom_list).to('cuda')
    atom_tens.unsqueeze_(0)
    
    
    return pos_tens, atom_tens

def ani_minimize_gpu(pos_tens,
                     atom_tens,
                     force,
                     model):
    """
    

    Parameters
    ----------
    pos_tens : Pytorch tensor
        A Pytorch tensor of shape [1, N_atoms, 3 (xyz)]
    atom_list : Pytorch tensor
        An ordered list of atom elemental numbers
    force : float
        The threshold force that stops minimization
    model : ANI-2x trained model
        ANI-2x pre-trained model provided from github

    Returns
    -------
    None.
    """
    
    #Create ANI-2x minimization calculator
    calixarene = ase.Atoms(atom_tens.cpu().squeeze().detach().numpy(),
                           positions=pos_tens.cpu().squeeze().detach().numpy())
    calc = torchani.ase.Calculator(calixarene, model)
    calixarene.set_calculator(calc)
    optimize = BFGS(calixarene)
    optimize.run(fmax=force)
        
    energy = model((atom_tens, pos_tens)).energies
    kcal = torchani.units.hartree2kcalmol(energy.item())
    
    return kcal

def ani_minimize_gpu_l(pos_tens,
                       atom_tens,
                       force,
                       model):
    """
    

    Parameters
    ----------
    pos_tens : Pytorch tensor
        A Pytorch tensor of shape [1, N_atoms, 3 (xyz)]
    atom_list : Pytorch tensor
        An ordered list of atom elemental numbers
    force : float
        The threshold force that stops minimization
    model : ANI-2x trained model
        ANI-2x pre-trained model provided from github

    Returns
    -------
    The energy in kcal of the minimized conformer, and the ase.Atoms object for
    use in updating RDKit conformer object
    """
    
    #Silence console output
    sys.stdout = open(os.devnull, "w")
    
    #Create ANI-2x minimization calculator
    heterocycle = ase.Atoms(atom_tens.cpu().squeeze().detach().numpy(),
                           positions=pos_tens.cpu().squeeze().detach().numpy())
    calc = torchani.ase.Calculator(heterocycle, model)
    heterocycle.set_calculator(calc)
    optimize = LBFGS(heterocycle)
    optimize.run(fmax=force)
        
    energy = model((atom_tens, pos_tens)).energies
    kcal = torchani.units.hartree2kcalmol(energy.item())
    
    #Return console
    sys.stdout = sys.__stdout__
    
    return kcal, heterocycle



def ani_analyze(pos_tens,
                 atom_tens,
                 model):
    """
    Uses ANI-2x energies to energy analyze a single RDKit Conformer.
    Returns the energy in kcal/mol (must convert from Hartree)

    Parameters
    ----------
    pos_tens : Pytorch tensor
        A Pytorch tensor of shape [1, N_atoms, 3 (xyz)]
    atom_list : list
        An ordered list of atom elemental numbers
    model : ANI-2x trained model
        ANI-2x pre-trained model provided from github

    Returns
    -------
    An energy in kcal/mol for a minimized conformer, as well as the ase.Atoms
    object: this object will have the new atom positions *not* the original conformer

    """
        
    energy = model((atom_tens, pos_tens)).energies
    kcal = 23.061 * energy.item() #Go from eV to kcal/mol
    
    return kcal
    
def gen_min_confs(molecule,
                  num_conformers,
                  force,
                  model):
    """
    A function that takes in an RDKit Molecule object, creates conformers (=conformers),
    and minimizes these conformers according to the loaded model

    Parameters
    ----------
    molecule : RDKit Molecule objet
        Actual calixarene being minimized
    conformers : integer
        Number of conformers to be generated and minimized
    force : float
        The threshold force that stops minimization
    model : ANI-2x trained model
        ANI-2x pre-trained model provided from github

    Returns
    -------
    A list of tuples. Each tuple is of the structure (conformer energy,
    ase.Atoms object that has .get_positions() function for coordinates)

    """    
    AllChem.EmbedMultipleConfs(molecule, pruneRmsThresh=0.2, numConfs=num_conformers)
    rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(molecule, maxIters=1250)
    return_list = []
    for conformer in molecule.GetConformers():
        pos_tens, atom_tens = conf_to_torch(molecule,
                                            conformer)
        kcal = ani_minimize_gpu(pos_tens,
                                atom_tens,
                                force,
                                model)
        return_list.append(kcal)
    rel_en_lis = [x - min(return_list) for x in return_list]
    return rel_en_lis

def gen_min_confs_l(molecule,
                    num_conformers,
                    force,
                    model):
    """
    A function that takes in an RDKit Molecule object, creates conformers (=conformers),
    and minimizes these conformers according to the loaded model

    Parameters
    ----------
    molecule : RDKit Molecule objet
        Actual calixarene being minimized
    conformers : integer
        Number of conformers to be generated and minimized
    force : float
        The threshold force that stops minimization
    model : ANI-2x trained model
        ANI-2x pre-trained model provided from github

    Returns
    -------
    A list of tuples. Each tuple is of the structure (conformer energy,
    ase.Atoms object that has .get_positions() function for coordinates)

    """    
    AllChem.EmbedMultipleConfs(molecule, pruneRmsThresh=0.2, numConfs=num_conformers)
    rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(molecule, maxIters=250)
    return_list = []
    for conformer in molecule.GetConformers():
        pos_tens, atom_tens = conf_to_torch(molecule,
                                            conformer)
        kcal, atom_object = ani_minimize_gpu_l(pos_tens,
                                               atom_tens,
                                               force,
                                               model)
        conf_set_positions(atom_object,
                           conformer)
        return_list.append([kcal, conformer])
    
    min_val_list = []
    for val_pair in return_list:
        min_val_list.append(val_pair[0])
    
    min_conf_energy = min(min_val_list)


    for entry in return_list:
        entry[0] = entry[0] - min_conf_energy
    return return_list

def conf_set_positions(atom_object,
                       conformer):
    """
    Takes an RDKit Conformer, and updates the atom positions to those that come
    from an ANI-2x minimization procedure

    Parameters
    ----------
    atom_object : .ase Atom type object that has been ANI-2x minimized
        .ase Atom type object that has been ANI-2x minimized
    conformer : RDKit conformer object
        The RDKit conformer object that supplied the initial atomic positions
        to the minimization script.

    Returns
    -------
    None. Moving between RDKit molecule,
    conformer, and ase.atoms does *not* change order of atoms (phew)

    """    
    
    ani_positions = atom_object.get_positions()
    for atom_idx in range(len(ani_positions)):
        x,y,z = ani_positions[atom_idx]
        conformer.SetAtomPosition(atom_idx, Point3D(x,y,z))
    
    return
    
    
#--------------------
#    Unit Tests    
#--------------------

def speed_test(molecule,
               force):
    """
    First test of CPU vs GPU speed for loading-minimizing molecules

    Parameters
    ----------
    molecule : An RDKit Molecule object
        Actual calixarene being analyzed
    force : float
        The threshold force that stops minimization

    Returns
    -------
    None - but time.time() is printed from function that calls this test

    """    
    sys.stdout = open(os.devnull, 'w')
    model = load_ani2()
    return_list = gen_min_confs(molecule,
                                25,
                                force,
                                model)
    sys.stdout = sys.__stdout__
    return_list.sort()
    print('For BFGS: ', return_list)
        
    return
    
def speed_test_l(molecule,
                 force):
    """
    First test of CPU vs GPU speed for loading-minimizing molecules, using LBFGS
    rather than BFGS

    Parameters
    ----------
    molecule : An RDKit Molecule object
        Actual calixarene being analyzed

    Returns
    -------
    None - but time.time() is printed from function that calls this test

    """    
    sys.stdout = open(os.devnull, 'w')
    model = load_ani2()
    return_list = gen_min_confs_l(molecule,
                                100,
                                force,
                                model)
    sys.stdout = sys.__stdout__
    return_list.sort()    
    print('For LBFGS: ', return_list)
    
    return

def force_energy_tradeoff(molecule,
                          model):
    """
    A function that takes in a molecule, embeds a single conformer, then
    looks at how the energy changes vs. residual force

    Parameters
    ----------
    molecule : An RDKit Molecule object
        Actual calixarene being analyzed
    model : ANI-2x trained model
        ANI-2x pre-trained model provided from github

    Returns
    -------
    None - but prints key information while running

    """
    
    AllChem.EmbedMultipleConfs(molecule, pruneRmsThresh=0.1, numConfs=1)
    rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(molecule, maxIters=1250)
    conformer = molecule.GetConformers()[0]
    
    cutoff_force = [0.01, 0.001, 0.0001, 0.00005]
    
    pos_tens, atom_tens = conf_to_torch(molecule,
                                        conformer)
    
    energy_list = []
            
    for force in cutoff_force:
        start_time = time.time()
        sys.stdout = open(os.devnull, 'w')

        kcal, atom_object = ani_minimize_gpu_l(pos_tens,
                                               atom_tens,
                                               force,
                                               model)
        sys.stdout = sys.__stdout__

        energy_list.append(kcal)
        print('Time for force of', force, 'was: {:.2f}s'.format(time.time() - start_time))
    rel_list = [x - min(energy_list) for x in energy_list]
    print('Relative energies were:', rel_list)
    
    return
    
    
        
    
    
    
    
    
    
    
    
    
