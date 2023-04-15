import os
import torch
import pickle
import numpy as np

from scipy import signal
from mmsdk import mmdatasdk as md
from torch.utils.data.dataset import Dataset

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    
############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################


class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data_noalign.pkl' )
        dataset = pickle.load(open(dataset_path, 'rb'))

        if "mosi" in data:
            DATASET = md.cmu_mosi
        elif "mosei" in data:
            DATASET = md.cmu_mosei
        self.train_split = DATASET.standard_folds.standard_train_fold
        self.dev_split = DATASET.standard_folds.standard_valid_fold
        self.test_split = DATASET.standard_folds.standard_test_fold

        # These are torch tensors
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()

        # Note: this is STILL an numpy array
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None

        self.data = data

        self.n_modalities = 3 # vision/ text/ audio
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))

        sentence = None
        if self.data == 'mosei_senti':
            file = open(os.path.join('/home/ICT2000/yin/emnlp/data/CMU-MOSEI/process/Transcript/Segmented/Combined', META[0]+'.txt'), 'r')
            sentences = file.readlines()
            for line in sentences:
                line = line[:-1].split('___')
                start = line[2]
                end = line[3]
                if float(start) == float(META[1]) and float(end) == float(META[2]):
                    sentence = line[4]
                    break
        else:
            vid = '_'.join(META[0].split('_')[:-1])
            if vid in self.train_split:
                split = 'train'
            elif vid in self.dev_split:
                split = 'val'
            elif vid in self.test_split:
                split = 'test'
            file = open(os.path.join('/home/ICT2000/yin/emnlp/data/CMU-MOSI/text', split, META[0]+'.txt'), 'r')
            sentence = file.readline()

        X = (index, self.text[index], self.audio[index], self.vision[index], sentence)
        Y = self.labels[index]

        return X, Y, META
