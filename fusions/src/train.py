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
    parser = argparse.ArgumentParser(description='Unimodal or Fusion Parser')
    parser.add_argument('--text_encoder', type=str, default='glove', help='Text encoder, choose from glove, bert, deberta and roberta')
    parser.add_argument('--audio_encoder', type=str, default='hubert', help='Text encoder, choose from glove, bert, deberta and roberta')
    parser.add_argument('--model', type=str, default='EarlyFusion', help='Text encoder, choose from glove, bert, deberta and roberta')
    parser.add_argument('--data', type=str, default='mosei', help='Dataset to MISA on, choose from mosi or mosei')

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
    args_dict['learning_rate'] = 1e-5
    main_save_dir = f'{PROJECT_DIR}/fusions/results'
    if args.model in ['EarlyFusion', 'LateFusion', 'TextClassifier']:
        save_dir = f'{main_save_dir}/{args.data}/{args.text_encoder}'
        if args.audio_encoder == 'hubert' and args.model in ['EarlyFusion', 'LateFusion']:
            save_dir += f'_{args.audio_encoder}'
    else:
        save_dir = f'{main_save_dir}/{args.data}'
        if args.audio_encoder == 'hubert' and args.model == 'AcousticClassifier':
            save_dir += f'_{args.audio_encoder}'
            args_dict['learning_rate'] = 1e-3

    # Setting the config for each stage
    train_config = get_config(mode='train', save_dir=save_dir, **args_dict)
    dev_config = get_config(mode='dev', save_dir=save_dir, **args_dict)
    test_config = get_config(mode='test', save_dir=save_dir, **args_dict)

    print(train_config)

    # Creating pytorch dataloaders
    train_data_loader = get_loader(train_config, shuffle = True)
    dev_data_loader = get_loader(dev_config, shuffle = False)
    test_data_loader = get_loader(test_config, shuffle = False)

    # Solver is a wrapper for model training and testing
    solver = Solver
    solver = solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=False)

    # Build the model
    solver.build(cuda=cuda)

    # Train the model (test scores will be returned based on dev performance)
    solver.train(cuda=cuda)
