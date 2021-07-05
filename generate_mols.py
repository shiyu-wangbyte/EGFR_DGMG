# -*- coding: utf-8 -*-
#--------------------------------------------#
#Generate molecules with pretrained model or fine-tuned model
#--------------------------------------------#

#!/usr/bin/python
import dgl
import os
import pickle
import dgllife
import rdkit
from dgllife.model import DGMG
import torch
from tqdm import tqdm
from dgllife.model import load_pretrained
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Generate molecules with pretrained model or fine-tuned model.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset', default='EGFR', help='dataset to use')
parser.add_argument('-p', '--path',default=None, help='Path of fine-tuned model')
parser.add_argument('-n', '--number',default=100, type=int ,help='The number of molecules you want to generate.')
parser.add_argument('-m', '--model', default='ZINC',choices=['CHEMBL', 'ZINC'], help='Pre-trained model to use, in [CHEMBL,ZINC]')
parser.add_argument('-o', '--optput-file', type=str, default='./output.csv',
                        help='Path to a file recording the generated molecules. ')
parser.add_argument('-s', '--set-file', type=str, default=None,
                        help='Path to setting file of training or fine-tuning model. ')
args = parser.parse_args()

#load setting
dataset=args.dataset
path=args.path
set_file=args.set_file
name_model=args.model


#build model and load parameter
path_to_atom_and_bond_types = '_'.join([args.dataset, 'atom_and_bond_types.pkl'])
if not os.path.exists(path_to_atom_and_bond_types):
    print('there is no such file:',path_to_atom_and_bond_types)
else:
    with open(path_to_atom_and_bond_types, 'rb') as f:
        type_info = pickle.load(f)
        atom_types = type_info['atom_types']
        bond_types = type_info['bond_types']

if args.set_file is not None:
    df=pd.read_csv(set_file,sep='\t',index_col=0)
    node_hidden_size=int(df.loc['node_hidden_size'])
    num_prop_rounds=int(df.loc['num_propagation_rounds'])
    dropout=float(df.loc['dropout'])
else:
    node_hidden_size=128
    num_prop_rounds=2
    dropout=0.2


if args.path is not None:
    model = DGMG(atom_types=atom_types,
            bond_types=bond_types,
            node_hidden_size=node_hidden_size,
            num_prop_rounds=num_prop_rounds,
            dropout=dropout)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    if args.model =='CHEMBL':
        model=load_pretrained('DGMG_CHEMBL_canonical')
    else:
        model=load_pretrained('DGMG_ZINC_canonical')

#generate molecules
smi_list=[]
for i in tqdm(range(args.number)):
    smi=model(rdkit_mol=True)
    smi_list.append(smi)

#save molecules
data=pd.DataFrame({'smi':smi_list})
data.to_csv(args.optput_file,index=None)
print('#----------------------------------------------------#')
print('The molecule(s) have been saved to:',args.optput_file)
