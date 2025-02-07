import torchquantum as tq
import torchquantum.functional as tqf
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from dataset import QDrugDataset
from models.model import *
from models.Cmodel import *
import argparse
import numpy as np
import sys
from args import config_parser
import os
from utils.eval_validity import *
from utils.eval_property import *


torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def reconstruct_func(x, args):

    for i in range(args.max_atoms):
        if x[7*i+3:7*i+7].sum() <= args.threshold:
            break
    num_vertices = i

    if num_vertices == 0 or num_vertices ==1:
        return None, None

    x = x.detach().cpu().numpy()
    min_val = np.array([-3.92718444, -4.54352093, -5.10957203])
    diff_minmax = 10.36
    atomic_type_dict = {0:6, 1:7, 2:8, 3:9, 4:16, 5:17, 6:36, 7:53, 8:5}
    atom_type = np.zeros((num_vertices), dtype=int)
    position = np.zeros((num_vertices, 3))
    
    for i in range(num_vertices):
        atom_type[i] = np.argmax(x[i*7+3:i*7+7])
        atom_type[i] = atomic_type_dict[atom_type[i]]
        position[i] = x[i*7:i*7+3] * 2 * diff_minmax + min_val

    return atom_type, position
    
if __name__ == '__main__':
    setup_seed(12345)
    args = config_parser().parse_args()
    def random_generation():
        model = DrugQAE(args, n_qbits=args.n_qbits, n_blocks=args.n_blocks).to(torch_device)
        raw_params_dict = torch.load('checkpoint/model.pth')
        raw_params_dict['qdev.states'] = torch.zeros((1,2,2,2,2,2,2,2), dtype=torch.complex64)

        model.load_state_dict(raw_params_dict)
        model.to(torch_device)
        model.eval()

        generate_num = 1000
        atom_type, positions = [], []
        for i in range(generate_num):
            print(i)
            z_sample = np.abs(np.random.randn(1, 2**(args.n_qbits - args.bottleneck_qbits)))
            z_sample = torch.tensor(z_sample).to(torch_device)
            z_sample /= (z_sample**2).sum()**0.5
            x_decoded = model.generate(z_sample)
            atom_type_one, position_one = reconstruct_func(x_decoded[0], args)
            # print(x_decoded[0])
            if atom_type_one is not None:
                atom_type.append(atom_type_one)
                positions.append(position_one)

        drug_dict = {'number':atom_type, 'position': positions}
        
        valid_ratio, unique_ratio, novel_ratio, bond_mmds, _, _ = calculate_metric(drug_dict)
        print(f"validity: {valid_ratio}, unique:{unique_ratio}, novelty:{novel_ratio}")
        for key, value in bond_mmds.items():
            print(f" bond mmd {key}: {value}")

        print('valid ratio: ', valid_ratio)
        print('unique ratio: ', unique_ratio)
        print('novel ratio: ', novel_ratio)

    random_generation()

    


    


    
