import torch
import torch.nn as nn
print(torch.cuda.is_available())

import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F
from typing import List
from dataset import QDrugDataset
from models.model import *
import argparse
import numpy as np
import sys
from args import config_parser
import os
from generate import reconstruct_func
from utils.eval_validity import calculate_metric
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, args, reduction='mean'):
        """
        :param alpha: list of weight coefficients for each class, where in a 3-class problem,
        the weight for class 0 is 0.2, class 1 is 0.3, and class 2 is 0.5.
        :param gamma: coefficient for hard example mining
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(args.atom_weight).to(torch_device)
        self.gamma = args.focal_gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # for each sample in the current batch, assign weight to each class, shape=(bs), one-dimensional vector
        pt = torch.gather(pred, dim=1, index=target.view(-1, 1))  # extract the log_softmax value at the position of class label for each sample, shape=(bs, 1)
        pt = pt.view(-1)  # reduce dimension, shape=(bs)
        ce_loss = -torch.log(pt)  # take the negative of the log_softmax, which is the cross entropy loss
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # calculate focal loss according to the formula and obtain the loss value for each sample, shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch_device)

def loss_func_qm9(x, y, num_vertices, focal_loss):
    data_vertices = x[:7*num_vertices].reshape(num_vertices, 7)
    recon_vertices = y[:7*num_vertices].reshape(num_vertices, 7)

    data_atom_type = torch.argmax(data_vertices[:, 3:7], dim=-1)
    recon_atom_type = recon_vertices[:, 3:7]
    constrained_loss = F.mse_loss((recon_atom_type ** 2).sum(dim=1), torch.tensor([0.25 / num_vertices] * num_vertices).to(torch_device))
    recon_atom_type = recon_vertices[:, 3:7] ** 2 * num_vertices * 2
    recon_atom_type = recon_atom_type / recon_atom_type.sum(dim=1).unsqueeze(1)
    

    atom_type_loss = focal_loss(recon_atom_type, data_atom_type)
    
    data_xyz = data_vertices[:, :3]
    recon_xyz = recon_vertices[:, :3]
    data_aux = x[7*num_vertices:8*num_vertices]

    recon_aux = y[7*num_vertices:8*num_vertices]
    recon_remain = y[8*num_vertices:]
    xyz_norm_val = torch.linalg.norm(recon_xyz-data_xyz, axis=-1).mean()

    data_atom_type = torch.argmax(data_vertices[:, 3:7], dim=-1)
    recon_atom_type = torch.argmax(recon_vertices[:, 3:7], dim=-1)

    accuracy = (data_atom_type == recon_atom_type).float().mean()
    return xyz_norm_val + torch.abs(recon_aux-data_aux).mean() + torch.abs(recon_remain).mean()+ constrained_loss*100 + atom_type_loss, \
        xyz_norm_val, torch.abs(recon_aux-data_aux).mean(), torch.abs(recon_remain).mean(), constrained_loss, atom_type_loss, accuracy

def fidelity_loss(x, y):
    return 1 - (torch.dot(x, y)) ** 2

def generate(model, args, epoch, min_val, diff_minmax, f):
    generate_num = 1000
    atom_type, positions = [], []
    for i in range(generate_num):
        z_sample = np.abs(np.random.randn(1, 2**(args.n_qbits - args.bottleneck_qbits)))
        z_sample = torch.tensor(z_sample).to(torch_device)
        z_sample /= (z_sample**2).sum()**0.5
        x_decoded = model.generate(z_sample)

        atom_type_one, position_one = reconstruct_func(x_decoded[0], args)
        if atom_type_one is not None:
            atom_type.append(atom_type_one)
            positions.append(position_one)

    drug_dict = {'number':atom_type, 'position': positions}
    
    valid_ratio, unique_ratio, novel_ratio, bond_mmds,_, _ = calculate_metric(drug_dict)

    print(f"Epoch {epoch} validity: {valid_ratio}, unique:{unique_ratio}, novelty:{novel_ratio}")
    f.write(f"Epoch {epoch} validity: {valid_ratio}, unique:{unique_ratio}, novelty:{novel_ratio}\n")

    if valid_ratio == 0:
        return

    for key, value in bond_mmds.items():
        print(f"Epoch {epoch} bond mmd {key}: {value}")
        f.write(f"Epoch {epoch} bond mmd {key}: {value}\n")


def main():

    setup_seed(0)

    parser = config_parser()
    args = parser.parse_args()
    if args.dataset == 'qm9':
        args.atom_weight = [0.3, 0.8, 0.5, 1.2]

    dataset = QDrugDataset(args, True)
    min_val, diff_minmax = dataset.min_val, dataset.diff_minmax

    train_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.RandomSampler(dataset),
    )

    model = DrugQAE(args, n_qbits=args.n_qbits, n_blocks=args.n_blocks).to(torch_device)
    print(args.n_qbits)
    model.to(torch_device)
    focal_loss = MultiClassFocalLossWithAlpha(args)

    losses = []
    epochs = args.num_epochs
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    for epoch in range(epochs):
        running_loss = 0.0
        batches = 0
        
        model.train()

        if args.dataset == 'qm9':
            loss_func = loss_func_qm9

        for batch_dict in train_dl:
            x_whole, smi = batch_dict['x'], batch_dict['smi']

            x = x_whole[:, :-1].to(torch_device)
            num_vertices = x_whole[:, -1].to(torch_device).int()

            measure, reconstruct = model(x)
            reconstruct = reconstruct.to(torch_device)

            loss, xyz_norm_val, atom_type_loss, aux_loss, remain_loss, constrained_loss, acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            for i in range(x.shape[0]):
                loss_, xyz_norm_val_, aux_loss_, \
                    remain_loss_, constrained_loss_, atom_type_loss_,\
                        acc_ = loss_func(x[i], reconstruct[i], num_vertices[i], focal_loss)
                loss += loss_
                xyz_norm_val += xyz_norm_val_
                aux_loss += aux_loss_
                remain_loss += remain_loss_
                constrained_loss += constrained_loss_
                atom_type_loss += atom_type_loss_
                acc += acc_
            
            sys.stdout.write(
                f"\r{batches} / {len(train_dl)}, "
                f"{(loss / x.shape[0]): .6f}, "
                f"{(xyz_norm_val / x.shape[0]): .6f}, "
                f"{(aux_loss / x.shape[0]): .6f}, "
                f"{(remain_loss / x.shape[0]): .6f}, "
                f"{(constrained_loss / x.shape[0]): .6f}, "
                f"{(atom_type_loss / x.shape[0]): .6f}, "
                f"{(acc / x.shape[0]): .6f}, "
                "          "
            )
            sys.stdout.flush()

            loss /= x.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1

        losses.append(running_loss / batches)
        print(f"Epoch {epoch} loss: {losses[-1]}")
        
        save_folder = f'./save/{args.dataset}_results/{args.n_qbits}_{args.n_blocks}_{args.kappa}_{args.lr}_{args.use_MLP}_{args.threshold}_{args.max_atoms}/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        torch.save(model.state_dict(), save_folder + f"model_{epoch}.pth")
        f = open(save_folder + f"res.txt", 'a')
        f.write(f"Epoch {epoch} loss: {losses[-1]}\n")
        model.eval()
        generate(model, args, epoch, min_val, diff_minmax, f)

if __name__ == '__main__':
    print(torch.cuda.is_available())
    main()




