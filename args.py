import argparse


def config_parser():
    parser = argparse.ArgumentParser(description='Qdrug')

    # training options
    parser.add_argument("--n_qbits", type=int, default=7, help='number of qubits in the QAE')
    parser.add_argument("--num_epochs", type=int, default=100, help='number of epochs to run')
    parser.add_argument("--lr", type=float, default=0.01, help='initial learning rate')
    parser.add_argument("--batch_size", type=int, default=512, help='batch size')
    parser.add_argument("--n_blocks", type=int, default=10, help='no of blocks in the QAE')
    parser.add_argument("--optim", type=str, default='adam', help='optimizer to use')
    parser.add_argument("--weight_decay", type=float, default=0.0001, help='weight decay')
    parser.add_argument("--focal_gamma", type=float, default=2, help='focal gamma')
    parser.add_argument("--atom_weight", type=list, default=[], help='atom weight')
    parser.add_argument("--use_MLP", type=bool, default=False, help='use MLP')
    parser.add_argument("--input_dim", type=int, default=73, help='input dim') # qm9 : 9 * (3+4+1), zinc : x * (3+9+1)
    parser.add_argument("--use_sigmoid", type=bool, default=False, help='use sigmoid')
    parser.add_argument("--dataset", type=str, default='qm9', help='dataset')
    parser.add_argument("--kappa", type=float, default=20, help='save dir')
    parser.add_argument("--bottleneck_qbits", type=int, default=2, help='save dir')
    parser.add_argument("--hidden_dim", type=int, default=512, help='save dir')
    parser.add_argument("--postprecess", type=int, default=0, help='save dir')
    parser.add_argument("--threshold", type=float, default=0.2, help='save dir')
    parser.add_argument("--max_atoms", type=int, default=12, help='max_atoms')
    parser.add_argument("--num_condition", type=int, default=0)
    parser.add_argument("--property_index", type=int, default=0)

    
    return parser