import torchquantum as tq
import torchquantum.functional as tqf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchquantum.operator import op_name_dict
from typing import List
from dataset import QDrugDataset
import math
import numpy as np


class DrugQAE(tq.QuantumModule):
    def __init__(self, args, n_qbits=10, n_blocks=1, bottleneck_qbits=2, use_sigmoid=False):
        super().__init__()
        self.args = args
        self.MLP_in = torch.nn.Linear(args.input_dim, 2 ** args.n_qbits)
        self.MLP_out = torch.nn.Linear(2 ** args.n_qbits, args.input_dim)
        self.encoder = tq.AmplitudeEncoder() # TODO: can change to multiphaseencoder
        self.use_sigmoid = use_sigmoid
        self.n_blocks = n_blocks
        self.n_qbits = n_qbits
        self.bottleneck_qbits = bottleneck_qbits
        self.kappa = args.kappa

        self.ry_layer, self.rz1_layer, self.rz2_layer, self.crx_layer = \
            tq.QuantumModuleList(),tq.QuantumModuleList(),tq.QuantumModuleList(),tq.QuantumModuleList()
        
        self.ry_layer_d, self.rz1_layer_d, self.rz2_layer_d, self.crx_layer_d = \
            tq.QuantumModuleList(),tq.QuantumModuleList(),tq.QuantumModuleList(),tq.QuantumModuleList()
        
        for k in range(n_blocks+1):
            for i in range(n_qbits):
                if k == n_blocks:
                    self.rz1_layer.append(tq.RZ(trainable=True, has_params=True))
                    self.ry_layer.append(tq.RY(trainable=True, has_params=True))
                    self.rz2_layer.append(tq.RZ(trainable=True, has_params=True))
                else:
                    self.rz1_layer.append(tq.RZ(trainable=True, has_params=True))
                    self.ry_layer.append(tq.RY(trainable=True, has_params=True))
                    self.rz2_layer.append(tq.RZ(trainable=True, has_params=True))
                    self.crx_layer.append(tq.CRX(trainable=True, has_params=True))

        for k in range(n_blocks+1):
            for i in range(n_qbits):
                if k == n_blocks:
                    self.rz1_layer_d.append(tq.RZ(trainable=True, has_params=True))
                    self.ry_layer_d.append(tq.RY(trainable=True, has_params=True))
                    self.rz2_layer_d.append(tq.RZ(trainable=True, has_params=True))
                else:
                    self.rz1_layer_d.append(tq.RZ(trainable=True, has_params=True))
                    self.ry_layer_d.append(tq.RY(trainable=True, has_params=True))
                    self.rz2_layer_d.append(tq.RZ(trainable=True, has_params=True))
                    self.crx_layer_d.append(tq.CRX(trainable=True, has_params=True))

        self.service = "TorchQuantum"
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.qdev = tq.QuantumDevice(n_qbits)

    def digits_to_binary_approx(self, x):
        return 1 / (1 + torch.exp(-50*(x-0.5)))
    
    def sampling_vmf(self, mu, kappa):

        dims = mu.shape[-1]

        epsilon = 1e-7
        x = np.arange(-1 + epsilon, 1, epsilon)
        # print(x)
        y = kappa * x + np.log(1 - x**2) * (dims - 3) / 2
        y = np.cumsum(np.exp(y - y.max()))
        y = y / y[-1]
        rand_nums = np.interp(torch.rand(10**6), y, x)

        W = torch.Tensor(rand_nums).to(mu.device)

        idx = torch.randint(0, 10**6, (mu.size(0), 1)).to(torch.int64).to(mu.device)

        w = torch.gather(W, 0, idx.squeeze()).unsqueeze(1)

        eps = torch.randn_like(mu)
        nu = eps - torch.sum(eps * mu, dim=1, keepdim=True) * mu
        nu = torch.nn.functional.normalize(nu, p=2, dim=-1)

        return w * mu + (1 - w**2)**0.5 * nu
    
    def forward(self, x, measure=False):

        bsz = x.shape[0]
        self.qdev.reset_states(bsz)

        if self.args.use_MLP:
            x = self.MLP_in(x)

        if self.encoder:
            self.encoder(self.qdev, x)

        # encode phase
        for k in range(self.n_blocks+1):
            for i in range(self.n_qbits):
                self.rz1_layer[k*self.n_qbits+i](self.qdev, wires=i)
                self.ry_layer[k*self.n_qbits+i](self.qdev, wires=i)
                self.rz2_layer[k*self.n_qbits+i](self.qdev, wires=i)
            if k != self.n_blocks:
                for i in range(self.n_qbits):
                    self.crx_layer[k*self.n_qbits+i](self.qdev, wires=[i, (i+1)%self.n_qbits])
            
        # measure phase
        if measure:
            meas = self.measure(self.qdev)
            # decode phase
            if self.use_sigmoid:
                meas = self.digits_to_binary_approx(meas)
                self.qdev.reset_states(bsz)
                for i in range(self.n_qbits):
                    tqf.rx(self.qdev, wires=i, params=math.pi*meas[:, i])
            else:
                self.qdev.reset_states(bsz)
                for i in range(self.n_qbits):
                    tqf.ry(self.qdev, wires=i, params=meas[:, i])
        else:
            output1 = self.qdev.get_states_1d().abs()
            # 2 ^ 7 -> 2^ 5
            output1 = output1**2
            output1 = output1.reshape(-1, pow(2, self.n_qbits - self.bottleneck_qbits), pow(2, self.bottleneck_qbits)).sum(axis=-1).sqrt()
            # print(output1)
            decoder_in = self.sampling_vmf(output1, self.kappa)
            # print(decoder_in)

            self.qdev.reset_states(bsz)
            self.encoder(self.qdev, decoder_in)
        
        for k in range(self.n_blocks, -1, -1):
            if k == self.n_blocks:
                for i in range(self.n_qbits):
                    self.rz2_layer_d[k*self.n_qbits+i](self.qdev, wires=i)
                    self.ry_layer_d[k*self.n_qbits+i](self.qdev, wires=i)
                    self.rz1_layer_d[k*self.n_qbits+i](self.qdev, wires=i)
            else:
                for i in range(self.n_qbits-1, -1, -1):
                    self.crx_layer_d[k*self.n_qbits+i](self.qdev, wires=[i, (i+1)%self.n_qbits])
                for i in range(self.n_qbits):
                    self.rz2_layer_d[k*self.n_qbits+i](self.qdev, wires=i)
                    self.ry_layer_d[k*self.n_qbits+i](self.qdev, wires=i)
                    self.rz1_layer_d[k*self.n_qbits+i](self.qdev, wires=i)

        reconstruct_vector = self.qdev.get_states_1d().abs()
        if self.args.use_MLP:
            reconstruct_vector = self.MLP_out(reconstruct_vector)
                
        return output1, reconstruct_vector
    

    def generate(self, x):
        bsz = x.shape[0]
        self.qdev.reset_states(bsz)

        self.encoder(self.qdev, x)

        for k in range(self.n_blocks, -1, -1):
            if k == self.n_blocks:
                for i in range(self.n_qbits):
                    self.rz2_layer_d[k*self.n_qbits+i](self.qdev, wires=i)
                    self.ry_layer_d[k*self.n_qbits+i](self.qdev, wires=i)
                    self.rz1_layer_d[k*self.n_qbits+i](self.qdev, wires=i)
            else:
                for i in range(self.n_qbits-1, -1, -1):
                    self.crx_layer_d[k*self.n_qbits+i](self.qdev, wires=[i, (i+1)%self.n_qbits])
                for i in range(self.n_qbits):
                    self.rz2_layer_d[k*self.n_qbits+i](self.qdev, wires=i)
                    self.ry_layer_d[k*self.n_qbits+i](self.qdev, wires=i)
                    self.rz1_layer_d[k*self.n_qbits+i](self.qdev, wires=i)

        generate_vector = self.qdev.get_states_1d().abs()
                
        return generate_vector
