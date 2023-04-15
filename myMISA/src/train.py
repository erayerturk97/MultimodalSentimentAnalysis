import argparse
import pickle
import numpy as np
from random import random

from config import get_config, activation_dict, PROJECT_DIR
from data_loader import get_loader
from solver import Solver

import torch
import torch.nn as nn
from torch.nn import functional as F


if __name__ == '__main__':
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser(description='MISA Parser')
    parser.add_argument('--text_encoder', type=str, default='glove', help='Text encoder, choose from glove, bert, deberta and roberta')
    parser.add_argument('--data', type=str, default='mosi', help='Dataset to MISA on, choose from mosi or mosei')
    parser.add_argument('--add_noise', required=False, default=False, action='store_true', help='Adds noise to representations')
    parser.add_argument('--noise_cov_scale', type=float, default=1, action='store_true', help='Noise covariance added to the representations')
    parser.add_argument('--add_noise_batch_per', type=float, default=0.3, action='store_true', help='Percentage of test batches which noise will be added')

    args = parser.parse_args()
    args_dict = vars(args)

    # Setting random seed
    random_name = str(random())
    random_seed = 336   
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    
    # gpu or cpu
    cuda = True

    # save dir 
    main_save_dir = f'{PROJECT_DIR}/myMISA/results'
    save_dir = f'{main_save_dir}/{args.data}/{args.text_encoder}'

    # Setting the config for each stage
    train_config = get_config(mode='train', save_dir=save_dir, **args_dict)
    dev_config = get_config(mode='dev', save_dir=save_dir, **args_dict)
    test_config = get_config(mode='test', save_dir=save_dir, **args_dict)

    print(train_config)

    # Creating pytorch dataloaders
    train_data_loader = get_loader(train_config, shuffle = True)
    dev_data_loader = get_loader(dev_config, shuffle = False)
    test_data_loader = get_loader(test_config, shuffle = True) # False

    # Solver is a wrapper for model training and testing
    solver = Solver
    solver = solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True)

    # Build the model
    solver.build(cuda=cuda)

    # Train the model (test scores will be returned based on dev performance)
    solver.train(cuda=cuda)
