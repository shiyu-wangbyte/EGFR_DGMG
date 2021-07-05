# -*- coding: utf-8 -*-
#--------------------------------------------#
#Preprocessing additional data
#--------------------------------------------#

#!/usr/bin/python
from utils import configure_new_dataset
import argparse
from utils import setup
import pickle
import rdkit
from rdkit import Chem

parser = argparse.ArgumentParser(description='Preprocessing additional data for DGMG model',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset',help='dataset to use')
parser.add_argument('-m', '--model',help='pre-trained model to use, in [CHEMBL,ZINC]')
parser.add_argument('-tf', '--train-file', type=str, default=None,
                        help='Path to a file with one SMILES a line for training data. ')
parser.add_argument('-vf', '--val-file', type=str, default=None,
                        help='Path to a file with one SMILES a line for validation data. ')
args = parser.parse_args()

dataset=args.dataset
train_file=args.train_file
val_file=args.val_file

if args.model == 'ChEMBL':
    # For new datasets, get_atom_and_bond_types can be used to
    # identify the atom and bond types in them.
    atom_types = ['O', 'Cl', 'C', 'S', 'F', 'Br', 'N']
    bond_types = [Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE]

elif args.model == 'ZINC':
    atom_types = ['Br', 'S', 'C', 'P', 'N', 'O', 'F', 'Cl', 'I']
    bond_types = [Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE]
else :
    print('Please check pre-trained model! Must in [CHEMBL,ZINC]')

path_to_atom_and_bond_types = '_'.join([args.dataset, 'atom_and_bond_types.pkl'])
with open(path_to_atom_and_bond_types, 'wb') as f:
    pickle.dump({'atom_types': atom_types, 'bond_types': bond_types}, f)

configure_new_dataset(dataset, train_file, val_file)
