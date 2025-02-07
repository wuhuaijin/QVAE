import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader
import numpy as np
import random
import math
import pickle
import math
import os
from args import config_parser

class QDrugDataset(torch.utils.data.Dataset):
    def __init__(self, args, load_from_cache=False, file_path='./data/'):
        self.args = args
        self.atom_dict = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8}
        self.atomic_num_to_type = {6:0, 7:1, 8:2, 9:3}

        if args.dataset == 'qm9':
            self.atom_types = 4

        if load_from_cache:
            self.raw_data = pickle.load(open(file_path + 'raw_information.pkl', 'rb'))
        else:
            if args.dataset == 'qm9':
                mols = Chem.SDMolSupplier(os.path.join(file_path, 'gdb9.sdf'), removeHs=True, sanitize=True)
                molecular_information = []
                for i, mol in enumerate(mols):
                    try:
                        n_atoms = mol.GetNumAtoms()
                        if n_atoms > 9:
                            continue
                        pos = mols.GetItemText(i).split('\n')[4:4+n_atoms]
                        pos = np.array([[float(x) for x in line.split()[:3]] for line in pos])
                        pos = self.determine_position(pos)
                        if np.isnan(pos).any():
                            continue
                        atom_type = np.array([self.atomic_num_to_type[atom.GetAtomicNum()] for atom in mol.GetAtoms()]).reshape(-1, 1)
                        molecular_information.append({'position': pos, 'atom_type': atom_type, 'smi': Chem.MolToSmiles(mol)})
                    except:
                        continue

                self.raw_data = molecular_information
                pickle.dump(molecular_information, open(file_path + 'raw_information.pkl', 'wb'))
                
        position = []
        for item in self.raw_data:
            if np.isnan(item['position']).any():
                print('fuck')
            position.append(item['position'])
        position = np.concatenate(position, axis=0)
        print(position.shape)
        min_val = np.min(position, axis=0)
        print(min_val)
        max_val = np.max(position, axis=0)
        print(max_val)
        diff_minmax = np.max(max_val-min_val)
        print(diff_minmax)

        self.dataset = np.zeros((len(self.raw_data), 2**self.args.n_qbits + 1))
        self.info = []
        max_len = 0
        for i,item in enumerate(self.raw_data):
            position = (item['position'] - min_val) / diff_minmax
            atom_type = np.eye(self.atom_types)[item['atom_type'].astype(int)].squeeze(1)
            aux_vec = []
            for j in range(len(position)):
                aux_vec.append(math.sqrt(3 - (position[j][0] ** 2 + position[j][1] ** 2 + position[j][2] ** 2)))
            aux_vec = np.array(aux_vec)
            tmp = np.concatenate((position, atom_type), axis=1).flatten()
            self.dataset[i][:len(tmp)] = tmp
            self.dataset[i][len(tmp):len(position)+len(tmp)] = aux_vec
            self.dataset[i] = self.dataset[i] / (2*math.sqrt(len(position)))
            self.dataset[i][-1] = len(position)
            if len(position) > max_len:
                max_len = len(position)
            self.info.append({'x': self.dataset[i], 'smi': item['smi']})

        self.diff_minmax = diff_minmax
        self.min_val = min_val

    def determine_position(self, x):
        
        def move_point_cloud(points):
            centroid = np.mean(points, axis=0) 
            translated_points = points - centroid  
            return translated_points

        def rotate_point_cloud(points):

            first_point = points[0]

            x, y, z = first_point
            sin_angle_1 = -x / np.sqrt(x ** 2 + z ** 2)
            cos_angle_1 = z / np.sqrt(x ** 2 + z ** 2)

            sin_angle_2 = y / np.sqrt(x ** 2 + y ** 2 + z ** 2)
            cos_angle_2 = np.sqrt(x ** 2 + z ** 2) / np.sqrt(x ** 2 + y ** 2 + z ** 2)

            rotation_matrix = np.array([[cos_angle_1, 0, sin_angle_1],
                                        [sin_angle_2 * sin_angle_1, cos_angle_2, -sin_angle_2 * cos_angle_1],
                                        [-cos_angle_2 * sin_angle_1, sin_angle_2, cos_angle_2 * cos_angle_1]])

            rotated_point_cloud = np.dot(rotation_matrix, points.T).T
            return rotated_point_cloud
        
        x = move_point_cloud(x)
        x = rotate_point_cloud(x)
        return x
            
    def __getitem__(self, index):
        return self.info[index]

    def __len__(self):
        return len(self.dataset)
    

    
