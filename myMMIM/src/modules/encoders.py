import torch
import torch.nn.functional as F
import time

from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel, AutoConfig

def add_noise(x, intens=1e-7):
    return x + torch.rand(x.size()) * intens




class LanguageEmbeddingLayer(nn.Module):
    """Embed input text with "glove" or "Bert"
    """
    def __init__(self, hp):
        super(LanguageEmbeddingLayer, self).__init__()
        
        self.hp = hp
        rnn = nn.LSTM if self.hp.text_rnn_type == 'lstm' else nn.GRU
        if self.hp.bert_model == "bert":
            bertconfig = AutoConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
            self.bertmodel = AutoModel.from_pretrained("bert-base-uncased", config=bertconfig)
        elif self.hp.bert_model == "roberta":
            bertconfig = AutoConfig.from_pretrained("roberta-large", output_hidden_states=True)
            self.bertmodel = AutoModel.from_pretrained("roberta-large", config=bertconfig)
        elif self.hp.bert_model == "deberta":
            bertconfig = AutoConfig.from_pretrained("microsoft/deberta-v3-large", output_hidden_states=True)
            self.bertmodel = AutoModel.from_pretrained("microsoft/deberta-v3-large", config=bertconfig)
        elif self.hp.bert_model == 'glove':
            self.embed = nn.Embedding(len(hp.word2id), hp.d_tin)
            self.trnn1 = rnn(hp.d_tin, hp.d_th, bidirectional=True, batch_first=True)
            self.trnn2 = rnn(2*hp.d_th, hp.d_th, bidirectional=True, batch_first=True)
        self.tlayer_norm = nn.LayerNorm((hp.d_th*2,))

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        sequence = torch.permute(sequence, (1, 0, 2))
        packed_sequence = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True)

        if self.hp.text_rnn_type == 'lstm':
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1, batch_first=True, total_length=sequence.shape[1])
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths.cpu(), batch_first=True)

        if self.hp.text_rnn_type == 'lstm':
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2
    

    def forward(self, sentences, bert_sent, bert_sent_type, bert_sent_mask, lengths):
        
        if self.hp.bert_model == 'bert':
            bert_output = self.bertmodel(input_ids=bert_sent,
                                            attention_mask=bert_sent_mask,
                                            token_type_ids=bert_sent_type)
            bert_output = bert_output[0]
            # masked mean
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
        elif self.hp.bert_model == 'roberta':
            bert_output = self.bertmodel(input_ids=bert_sent,
                                            attention_mask=bert_sent_mask)
            bert_output = bert_output[0]
            # masked mean
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
        elif self.hp.bert_model == 'deberta':
            bert_output = self.bertmodel(input_ids=bert_sent,
                                            attention_mask=bert_sent_mask)
            bert_output = bert_output[0]
            # masked mean
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
        elif self.hp.bert_model == 'glove':
            # extract features from text modality
            sentences = self.embed(sentences)
            final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
            bert_output = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(lengths.size(0), -1)
        else:
            bert_output = self.bertmodel(input_ids=bert_sent,
                                         attention_mask=bert_sent_mask)
            bert_output = bert_output[0]
            # masked mean
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
        
        return bert_output   # return head (sequence representation)




















class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, n_class, dropout, modal_name='text'):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        # self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # normed = self.norm(x)
        dropped = self.drop(x)
        y_1 = torch.tanh(self.linear_1(dropped))
        fusion = self.linear_2(y_1)
        y_2 = torch.tanh(self.linear_2(y_1))
        y_3 = self.linear_3(y_2)
        return y_2, y_3

class CLUB(nn.Module):
    """
        Compute the Contrastive Log-ratio Upper Bound (CLUB) given a pair of inputs.
        Refer to https://arxiv.org/pdf/2006.12013.pdf and https://github.com/Linear95/CLUB/blob/f3457fc250a5773a6c476d79cda8cb07e1621313/MI_DA/MNISTModel_DANN.py#L233-254

        Args:
            hidden_size(int): embedding size
            activation(int): the activation function in the middle layer of MLP
    """
    def __init__(self, hidden_size, activation='Tanh'):
        super(CLUB, self).__init__()
        try:
            self.activation = getattr(nn, activation)
        except:
            raise ValueError("Error: CLUB activation function not found in torch library")
        self.mlp_mu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            self.activation(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            self.activation(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, modal_a, modal_b, sample=False):
        """
            CLUB with random shuffle, the Q function in original paper:
                CLUB = E_p(x,y)[log q(y|x)]-E_p(x)p(y)[log q(y|x)]
            
            Args:
                modal_a (Tensor): x in above equation
                model_b (Tensor): y in above equation
        """
        mu, logvar = self.mlp_mu(modal_a), self.mlp_logvar(modal_a) # (bs, hidden_size)
        batch_size = mu.size(0)
        pred = mu

        # pred b using a
        pred_tile = mu.unsqueeze(1).repeat(1, batch_size, 1)   # (bs, bs, emb_size)
        true_tile = pred.unsqueeze(0).repeat(batch_size, 1, 1)      # (bs, bs, emb_size)

        positive = - (mu - modal_b) ** 2 /2./ torch.exp(logvar)
        negative = - torch.mean((true_tile-pred_tile)**2, dim=1)/2./torch.exp(logvar)

        lld = torch.mean(torch.sum(positive, -1))
        bound = torch.mean(torch.sum(positive, -1)-torch.sum(negative, -1))
        return lld, bound

class MMILB(nn.Module):
    """Compute the Modality Mutual Information Lower Bound (MMILB) given bimodal representations.
    Args:
        x_size (int): embedding size of input modality representation x
        y_size (int): embedding size of input modality representation y
        mid_activation(int): the activation function in the middle layer of MLP
        last_activation(int): the activation function in the last layer of MLP that outputs logvar
    """
    def __init__(self, x_size, y_size, mid_activation='ReLU', last_activation='Tanh'):
        super(MMILB, self).__init__()
        try:
            self.mid_activation = getattr(nn, mid_activation)
            self.last_activation = getattr(nn, last_activation)
        except:
            raise ValueError("Error: CLUB activation function not found in torch library")
        self.mlp_mu = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size)
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size),
            nn.Tanh()
        )
        self.entropy_prj = nn.Sequential(
            nn.Linear(y_size, y_size // 4),
            nn.Tanh()
        )

    def forward(self, x, y, labels=None, mem=None):
        """ Forward lld (gaussian prior) and entropy estimation, partially refers the implementation
        of https://github.com/Linear95/CLUB/blob/master/MI_DA/MNISTModel_DANN.py
            Args:
                x (Tensor): x in above equation, shape (bs, x_size)
                y (Tensor): y in above equation, shape (bs, y_size)
        """
        mu, logvar = self.mlp_mu(x), self.mlp_logvar(x) # (bs, hidden_size)
        batch_size = mu.size(0)

        positive = -(mu - y)**2/2./torch.exp(logvar)
        lld = torch.mean(torch.sum(positive,-1))

        # For Gaussian Distribution Estimation
        pos_y = neg_y = None
        H = 0.0
        sample_dict = {'pos':None, 'neg':None}

        if labels is not None:
            # store pos and neg samples
            y = self.entropy_prj(y) 
            pos_y = y[labels.squeeze() > 0]
            neg_y = y[labels.squeeze() < 0]

            sample_dict['pos'] = pos_y
            sample_dict['neg'] = neg_y

            # estimate entropy
            if mem is not None and mem.get('pos', None) is not None:
                pos_history = mem['pos']
                neg_history = mem['neg']

                # Diagonal setting            
                # pos_all = torch.cat(pos_history + [pos_y], dim=0) # n_pos, emb
                # neg_all = torch.cat(neg_history + [neg_y], dim=0)
                # mu_pos = pos_all.mean(dim=0)
                # mu_neg = neg_all.mean(dim=0)

                # sigma_pos = torch.mean(pos_all ** 2, dim = 0) - mu_pos ** 2 # (embed)
                # sigma_neg = torch.mean(neg_all ** 2, dim = 0) - mu_neg ** 2 # (embed)
                # H = 0.25 * (torch.sum(torch.log(sigma_pos)) + torch.sum(torch.log(sigma_neg)))

                # compute the entire co-variance matrix
                pos_all = torch.cat(pos_history + [pos_y], dim=0) # n_pos, emb
                neg_all = torch.cat(neg_history + [neg_y], dim=0)
                mu_pos = pos_all.mean(dim=0)
                mu_neg = neg_all.mean(dim=0)
                sigma_pos = torch.mean(torch.bmm((pos_all-mu_pos).unsqueeze(-1), (pos_all-mu_pos).unsqueeze(1)), dim=0)
                sigma_neg = torch.mean(torch.bmm((neg_all-mu_neg).unsqueeze(-1), (neg_all-mu_neg).unsqueeze(1)), dim=0)
                H = 0.25 * (torch.logdet(sigma_pos) + torch.logdet(sigma_neg))

        return lld, sample_dict, H

class CPC(nn.Module):
    """
        Contrastive Predictive Coding: score computation. See https://arxiv.org/pdf/1807.03748.pdf.

        Args:
            x_size (int): embedding size of input modality representation x
            y_size (int): embedding size of input modality representation y
    """
    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)
        if n_layers == 1:
            self.net = nn.Linear(
                in_features=y_size,
                out_features=x_size
            )
        else:
            net = []
            for i in range(n_layers):
                if i == 0:
                    net.append(nn.Linear(self.y_size, self.x_size))
                    net.append(self.activation())
                else:
                    net.append(nn.Linear(self.x_size, self.x_size))
            self.net = nn.Sequential(*net)
        
    def forward(self, x, y):
        """Calulate the score 
        """
        # import ipdb;ipdb.set_trace()
        x_pred = self.net(y)    # bs, emb_size

        # normalize to unit sphere
        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True)

        pos = torch.sum(x*x_pred, dim=-1)   # bs
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)   # bs
        nce = -(pos - neg).mean()
        return nce

class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear((2 if bidirectional else 1)*hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        lengths = lengths.to(torch.int64)
        bs = x.size(0)

        packed_sequence = pack_padded_sequence(x, lengths, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        
        if self.bidirectional:
            h = self.dropout(torch.cat((final_states[0][0],final_states[0][1]),dim=-1))
        else:
            h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1
