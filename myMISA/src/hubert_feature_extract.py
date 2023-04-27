import argparse
import pickle
import numpy as np
from random import random

from config import get_config, activation_dict, PROJECT_DIR
from data_loader import MSADataset
from mmsdk import mmdatasdk as md
from solver import Solver
from scipy.io import wavfile
import os, re
from transformers import Wav2Vec2FeatureExtractor
from transformers import HubertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn
from torch.nn import functional as F


if __name__ == '__main__':

    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser(description='MISA Parser')
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

    # Creating pytorch dataloaders
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-er")
    model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-er")
    model.config.mask_time_length = 2 # raises error for embeddings of short audio 

    if cuda:
        model.cuda()
    
    os.makedirs(f'{PROJECT_DIR}/datasets/{args.data.upper()}/audio/hubert/train', exist_ok=True)
    os.makedirs(f'{PROJECT_DIR}/datasets/{args.data.upper()}/audio/hubert/val', exist_ok=True)
    os.makedirs(f'{PROJECT_DIR}/datasets/{args.data.upper()}/audio/hubert/test', exist_ok=True)
    os.makedirs(f'{PROJECT_DIR}/datasets/{args.data.upper()}/audio/hubert_all/train', exist_ok=True)
    os.makedirs(f'{PROJECT_DIR}/datasets/{args.data.upper()}/audio/hubert_all/val', exist_ok=True)
    os.makedirs(f'{PROJECT_DIR}/datasets/{args.data.upper()}/audio/hubert_all/test', exist_ok=True)

    audio_main_path = f'{PROJECT_DIR}/datasets/{args.data.upper()}/audio'

    for train_val in ['train', 'val', 'test']:
        if args.data == 'mosi':
            file_names = os.listdir(f'{audio_main_path}/{train_val}')
            for file in file_names:
                audio = wavfile.read(f'{audio_main_path}/{train_val}/{file}')[1]
                audio_norm = audio / 2**15

                input = feature_extractor(audio_norm, sampling_rate=16000, padding=True, return_tensors="pt")
                hubert_feat = torch.FloatTensor(input["input_values"])
                hubert_feats_att_mask = input["attention_mask"].to(torch.int)

                if cuda:
                    hubert_feat= hubert_feat.cuda()
                    hubert_feats_att_mask = hubert_feats_att_mask.cuda()

                try:
                    with torch.no_grad():
                        hubert_output = model(hubert_feat, hubert_feats_att_mask)
                        hubert_output = hubert_output.hidden_states[-1] 
                        hubert_output_mean = torch.mean(hubert_output, dim=1, keepdim=False) 
                except Exception as e:
                    print(e)
                    hubert_output = torch.zeros((1, 1, 768), dtype=torch.float32)
                    hubert_output_mean = torch.zeros((1, 768), dtype=torch.float32)

                torch.save(hubert_output_mean.detach().cpu(), f'{audio_main_path}/hubert/{train_val}/{file[:-4]}.pt')
                torch.save(hubert_output.detach().cpu(), f'{audio_main_path}/hubert_all/{train_val}/{file[:-4]}.pt')

        elif args.data == 'mosei':
            folder_names = os.listdir(f'{audio_main_path}/{train_val}')
            for folder in folder_names:
                folder_path = f'{audio_main_path}/{train_val}/{folder}'
                file_names = os.listdir(f'{audio_main_path}/{train_val}/{folder}')
                for file in file_names:
                    audio = wavfile.read(f'{audio_main_path}/{train_val}/{folder}/{file}')[1]
                    audio_norm = audio / 2**15

                    input = feature_extractor(audio_norm, sampling_rate=16000, padding=True, return_tensors="pt")
                    hubert_feat = torch.FloatTensor(input["input_values"])
                    hubert_feats_att_mask = input["attention_mask"].to(torch.int)

                    if cuda:
                        hubert_feat= hubert_feat.cuda()
                        hubert_feats_att_mask = hubert_feats_att_mask.cuda()

                    try:
                        with torch.no_grad():
                            hubert_output = model(hubert_feat, hubert_feats_att_mask)
                            hubert_output = hubert_output.hidden_states[-1] 
                            hubert_output_mean = torch.mean(hubert_output, dim=1, keepdim=False) 
                    except Exception as e:
                        print(e)
                        hubert_output = torch.zeros((1, 1, 768), dtype=torch.float32)
                        hubert_output_mean = torch.zeros((1, 768), dtype=torch.float32)

                    os.makedirs(f'{audio_main_path}/hubert/{train_val}/{folder}', exist_ok=True)
                    os.makedirs(f'{audio_main_path}/hubert_all/{train_val}/{folder}', exist_ok=True)

                    torch.save(hubert_output_mean.detach().cpu(), f'{audio_main_path}/hubert/{train_val}/{folder}/{file[:-4]}.pt')
                    torch.save(hubert_output.detach().cpu(), f'{audio_main_path}/hubert_all/{train_val}/{folder}/{file[:-4]}.pt')
