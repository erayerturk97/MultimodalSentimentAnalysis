import os, sys
import re
import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor, AutoProcessor
from scipy.io import wavfile
from config import project_dir

sys.path.append( r"C:\Users\slfgh\cs_535_project\CMU-MultimodalSDK" )
sys.path.append( r"/scratch1/yciftci/cs535project/CMU-MultimodalSDK" )

from mmsdk import mmdatasdk as md
from create_dataset import MOSI, MOSEI, PAD, UNK

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)   # burasi erayin dataloaderindan farkli
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-large")
deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
hubert_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")

class MSADataset(Dataset):
    def __init__(self, config):
        self.config = config

        ## Fetch dataset
        if "mosi" in str(config.data_dir).lower():
            dataset = MOSI(config)
        elif "mosei" in str(config.data_dir).lower():
            dataset = MOSEI(config)
        else:
            print("Dataset not defined correctly")
            exit()
        
        self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode)
        self.len = len(self.data)

        config.visual_size = self.data[0][0][1].shape[1]
        config.acoustic_size = self.data[0][0][2].shape[1]

        config.word2id = self.word2id
        config.pretrained_emb = self.pretrained_emb

    @property
    def tva_dim(self):
        if self.config.bert_model == "bert":
            t_dim = 768
        elif self.config.bert_model == "roberta":
            t_dim = 1024
        elif self.config.bert_model == "deberta":
            t_dim = 1024
        elif self.config.bert_model == "glove":
            t_dim = 512 #TODO
            print('----------------------------------------------')
            print('----------------------------------------------')
            print('----------------------------------------------')
            print('config.bert_model == "glove" and def tva_dim(self): is used ')
            print('----------------------------------------------')
            print('----------------------------------------------')
            print('----------------------------------------------')
        return t_dim, self.data[0][0][1].shape[1], self.data[0][0][2].shape[1]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def get_loader(hp, config, use_cuda, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)

    if "mosi" in str(config.data_dir).lower():
        DATASET = md.cmu_mosi
    elif "mosei" in str(config.data_dir).lower():
        DATASET = md.cmu_mosei
    train_split = DATASET.standard_folds.standard_train_fold
    dev_split = DATASET.standard_folds.standard_valid_fold
    test_split = DATASET.standard_folds.standard_test_fold
    
    print(config.mode)
    config.data_len = len(dataset)
    config.tva_dim = dataset.tva_dim
    
    if config.mode == "train":
        hp.n_train = len(dataset)
    elif config.mode == "dev":
        hp.n_valid = len(dataset)
    elif config.mode == "test":
        hp.n_test = len(dataset)

    def collate_fn(batch):
        """
        Collate functions assume batch = [Dataset[i] for i in index_set]
        """
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: len(x[0][3]), reverse=True)

        v_lens = []
        a_lens = []
        labels = []
        ids = []

        for sample in batch:
            if len(sample[0]) > 4: # unaligned case
                v_lens.append(torch.IntTensor([sample[0][4]]))
                a_lens.append(torch.IntTensor([sample[0][5]]))
            else:   # aligned cases
                v_lens.append(torch.IntTensor([len(sample[0][3])]))
                a_lens.append(torch.IntTensor([len(sample[0][3])]))
            labels.append(torch.from_numpy(sample[1]))
            ids.append(sample[2])
        vlens = torch.cat(v_lens)
        alens = torch.cat(a_lens)
        labels = torch.cat(labels, dim=0)
        
        # MOSEI sentiment labels locate in the first column of sentiment matrix
        if labels.size(1) == 7:
            labels = labels[:,0][:,None]

        # Rewrite this
        def pad_sequence(sequences, target_len=-1, batch_first=False, padding_value=0.0):
            if target_len < 0:
                max_size = sequences[0].size()
                trailing_dims = max_size[1:]
            else:
                max_size = target_len
                trailing_dims = sequences[0].size()[1:]

            max_len = max([s.size(0) for s in sequences])
            if batch_first:
                out_dims = (len(sequences), max_len) + trailing_dims
            else:
                out_dims = (max_len, len(sequences)) + trailing_dims

            out_tensor = sequences[0].new_full(out_dims, padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                if batch_first:
                    out_tensor[i, :length, ...] = tensor
                else:
                    out_tensor[:length, i, ...] = tensor
            return out_tensor

        sentences = pad_sequence([torch.tensor(sample[0][0], dtype=torch.int32) for sample in batch],padding_value=PAD)
        visual = pad_sequence([torch.tensor(sample[0][1], dtype=torch.float32) for sample in batch], target_len=vlens.max().item())
        acoustic = pad_sequence([torch.tensor(sample[0][2], dtype=torch.float32) for sample in batch],target_len=alens.max().item())

        if config.text_encoder in ['bert', 'glove']:
            SENT_LEN = sentences.size(0)
            # Create bert indices using tokenizer
            bert_details = []
            for sample in batch:
                text = " ".join(sample[0][3])
                encoded_bert_sent = bert_tokenizer.encode_plus(
                    text, max_length=SENT_LEN+2, add_special_tokens=True, truncation=True, padding="max_length")
                bert_details.append(encoded_bert_sent)

            # Bert things are batch_first
            bert_sentences = torch.tensor([sample["input_ids"] for sample in bert_details], dtype=torch.int32)
            bert_sentence_types = torch.tensor([sample["token_type_ids"] for sample in bert_details], dtype=torch.int32)
            bert_sentence_att_mask = torch.tensor([sample["attention_mask"] for sample in bert_details], dtype=torch.int32)

        elif config.text_encoder == 'roberta':
            text_list = []
            for sample in batch:
                text = " ".join(sample[0][3])
                text_list.append(text)
            encoded_bert_sent = roberta_tokenizer(text_list, padding=True, truncation=True,
                                            max_length=roberta_tokenizer.model_max_length, return_tensors="pt")

            bert_sentences = encoded_bert_sent["input_ids"].to(torch.int)
            bert_sentence_types = bert_sentences
            bert_sentence_att_mask = encoded_bert_sent["attention_mask"].to(torch.int)
        elif config.text_encoder == 'deberta':
            text_list = []
            for sample in batch:
                text = " ".join(sample[0][3])
                text_list.append(text)
            encoded_bert_sent = deberta_tokenizer(text_list, padding=True, truncation=True,
                                            max_length=roberta_tokenizer.model_max_length, return_tensors="pt")

            bert_sentences = encoded_bert_sent["input_ids"].to(torch.int) # torch.tensor(encoded_bert_sent["input_ids"])
            bert_sentence_types = bert_sentences
            bert_sentence_att_mask = encoded_bert_sent["attention_mask"].to(torch.int) #  torch.tensor(encoded_bert_sent["attention_mask"])







        # get raw audio
        audio_list = []
        if "mosi" in str(config.data_dir).lower():
            for sample in batch:
                segment = sample[2]
                pattern = re.compile('(.*)\[.*\]')
                vid = re.search(pattern, segment).group(1)
                idx = str(int(segment.replace(vid, '')[1:-1])+1)
                if vid in train_split:
                    split = 'train'
                elif vid in dev_split:
                    split = 'val'
                elif vid in test_split:
                    split = 'test'
                audio = wavfile.read(os.path.join(f'{project_dir}/datasets/MOSI/audio', split, vid+'_'+idx+'.wav'))[1]
                audio_norm = audio / 2**15
                audio_list.append(audio_norm)
        elif "mosei" in str(config.data_dir).lower():
            for sample in batch:
                segment = sample[2]
                pattern = re.compile('(.*)\[.*\]')
                vid = re.search(pattern, segment).group(1)
                if vid in train_split:
                    split = 'train'
                elif vid in dev_split:
                    split = 'val'
                elif vid in test_split:
                    split = 'test'
                idx = int(segment.replace(vid, '')[1:-1])
                file_list = sorted(os.listdir(os.path.join(f'{project_dir}/datasets/MOSEI/audio', split, vid)))
                audio = wavfile.read(os.path.join(f'{project_dir}/datasets/MOSEI/audio', split, vid, file_list[idx]))[1]
                audio_norm = audio / 2**15
                audio_list.append(audio_norm)

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-er")
        inputs = feature_extractor(audio_list, sampling_rate=16000, padding=True, return_tensors="pt")
        
        hubert_feats = torch.tensor(inputs["input_values"])
        hubert_feats_att_mask = inputs["attention_mask"].to(torch.int)

        # get hubert embeddings
        hubert_embedding_list = []
        if "mosi" in str(config.data_dir).lower():
            for sample in batch:
                segment = sample[2]
                pattern = re.compile('(.*)\[.*\]')
                vid = re.search(pattern, segment).group(1)
                idx = str(int(segment.replace(vid, '')[1:-1])+1)
                if vid in train_split:
                    split = 'train'
                elif vid in dev_split:
                    split = 'val'
                elif vid in test_split:
                    split = 'test'
                path = os.path.join(f'{project_dir}/datasets/MOSI/audio/hubert', split, vid+'_'+idx+'.pt')
                hubert_embedding = torch.load(path)
                hubert_embedding_list.append(hubert_embedding)

        elif "mosei" in str(config.data_dir).lower():
            for sample in batch:
                segment = sample[2]
                pattern = re.compile('(.*)\[.*\]')
                vid = re.search(pattern, segment).group(1)
                if vid in train_split:
                    split = 'train'
                elif vid in dev_split:
                    split = 'val'
                elif vid in test_split:
                    split = 'test'
                idx = int(segment.replace(vid, '')[1:-1])
                file_list = sorted(os.listdir(os.path.join(f'{project_dir}/datasets/MOSEI/audio/hubert', split, vid)))
                path = os.path.join(f'{project_dir}/datasets/MOSEI/audio/hubert', split, vid, file_list[idx])
                hubert_embedding = torch.load(path)
                hubert_embedding_list.append(hubert_embedding)

        hubert_embeddings = torch.cat(hubert_embedding_list, dim=0).detach()

        # lengths are useful later in using RNNs
        lengths = torch.tensor([sample[0][0].shape[0] for sample in batch], dtype=torch.int32)

        # # batch first for distributed training
        # sentences = torch.permute(sentences, (1,0))
        # visual = torch.permute(visual, (1,0,2))
        # acoustic = torch.permute(acoustic, (1,0,2))









        # lengths are useful later in using RNNs
        if (vlens <= 0).sum() > 0:
            vlens[np.where(vlens == 0)] = 1

        return sentences, visual, vlens, acoustic, alens, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask, ids, hubert_feats, hubert_feats_att_mask, hubert_embeddings
        

    if torch.cuda.is_available() and use_cuda:
        generator = torch.Generator(device='cuda')
    else:
        generator = torch.Generator(device='cpu')
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=generator)

    return data_loader

