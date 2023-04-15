import torch
import argparse
import numpy as np
import os, sys
from utils import *
from torch.utils.data import DataLoader
from solver import Solver
from solver_text import Solver_Text
from solver_fusion import Solver_Fusion
from solver_gb import Solver_GB
from config import get_args, get_config, output_dim_dict, criterion_dict, project_dir
from data_loader import get_loader
import pickle

def set_seed(seed, use_cuda):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(seed)
    if torch.cuda.is_available() and use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    print('Cuda available:', torch.cuda.is_available())
    args = get_args()
    use_cuda = True

    
    # args.test = True

    if "win" in sys.platform:
        # args.text_encoder = 'roberta'  # glove, bert, deberta and roberta
        # args.fusion = 'gb'
        # use_cuda = True
        # args.dataset = 'mosei'
        pass

    


    dataset = str.lower(args.dataset.strip())
    args.bert_model = args.text_encoder
    
    # if dataset == 'mosei' and args.text_encoder == 'deberta':
    #     args.batch_size = 16

    # save dir 
    main_save_dir = f'{project_dir}/r'
    save_dir = f'{main_save_dir}/{args.dataset}/{args.fusion}/{args.text_encoder}'
    args.save_dir = save_dir

    print('---------------------------------------')
    print('---------------------------------------')
    print( 'SAVE DIR:', save_dir)
    print('---------------------------------------')
    print('---------------------------------------')
    print(args)
    print('---------------------------------------')
    print('---------------------------------------')


    set_seed(args.seed, use_cuda=use_cuda)
    print("Start loading the data....")
    if args.test:
        args.num_epochs = 0

    train_config = get_config(dataset, mode='train', batch_size=args.batch_size, bert_model=args.bert_model, text_encoder=args.text_encoder)
    valid_config = get_config(dataset, mode='dev', batch_size=args.batch_size, bert_model=args.bert_model, text_encoder=args.text_encoder)
    test_config = get_config(dataset, mode='test',  batch_size=args.batch_size, bert_model=args.bert_model, text_encoder=args.text_encoder)

    # pretrained_emb saved in train_config here
    train_loader = get_loader(args, train_config, use_cuda= use_cuda, shuffle=True)
    print('Training data loaded!')
    valid_loader = get_loader(args, valid_config, use_cuda= use_cuda, shuffle=False)
    print('Validation data loaded!')
    test_loader = get_loader(args, test_config, use_cuda= use_cuda, shuffle=False)
    print('Test data loaded!')
    print('Finish loading the data....')

    torch.autograd.set_detect_anomaly(True)

    # addintional appending
    args.word2id = train_config.word2id

    # architecture parameters
    args.d_tin, args.d_vin, args.d_ain = train_config.tva_dim
    args.dataset = args.data = dataset
    args.when = args.when
    args.n_class = output_dim_dict.get(dataset, 1)
    args.criterion = criterion_dict.get(dataset, 'MSELoss')
    

    if args.fusion == 'none': # MMIM
        solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader, is_train=True, use_cuda=use_cuda)
    elif args.fusion == 'text': # bert
        solver = Solver_Text(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader, is_train=True)
    elif args.fusion == 'early' or args.fusion == 'late': # early or late fusion
        solver = Solver_Fusion(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader, is_train=True)
    elif args.fusion == 'audio' or args.fusion == 'video': # audio or video uni-modal
        solver = Solver_Fusion(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader, is_train=True)
    elif args.fusion == 'gb': # gradient blending
        solver = Solver_GB(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader, is_train=True, use_cuda=use_cuda)
    
    if not args.test:
        result_dict = solver.train_and_eval()
    else:
        result_dict = solver.test()
    
    # save dict to save dir    
    save_results_directory = f'{args.save_dir}/test_results'
    save_results_path = save_results_directory + '/results.pickle'

    if not os.path.exists(save_results_directory):
        os.makedirs(save_results_directory)    

    if not os.path.isfile(save_results_path):
        pickle.dump(result_dict, open(f'{save_results_path}', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        print('RESULTS ARE SAVED at: ', save_results_path)
    else:
        print('this file already exists: {}'.format(save_results_path))


    print('---------------------------')
    print('--------- FINISH   ---------------')
    print('--------------------------------')





# with open(r'C:\Users\slfgh\cs_535_project\myMMIM\r\mosi\none\bert\test_results\results.pickle', 'rb') as handle:
#     b = pickle.load(handle)