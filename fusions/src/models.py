import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import AutoModel, AutoConfig


def masked_mean(tensor, mask, dim):
    #Finding the mean along dim
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)

def masked_max(tensor, mask, dim):
    #Finding the max along dim
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)


class TextClassifier(nn.Module):
    def __init__(self, config):
        super(TextClassifier, self).__init__()

        self.config = config

        self.input_size = config.embedding_size
        self.hidden_size = int(config.embedding_size)
        self.output_size = config.num_classes
        self.dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()

        rnn = nn.LSTM if self.config.rnncell == 'lstm' else nn.GRU

        # defining modules - two layer bidirectional LSTM with layer norm in between
        if self.config.text_encoder == 'glove':
            self.embed = nn.Embedding(len(config.word2id), self.input_size)
            self.rnn1 = rnn(self.input_size, self.hidden_size, bidirectional=True, batch_first=True)
            self.rnn2 = rnn(2*self.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        elif self.config.text_encoder == 'bert':
            # Initializing a BERT bert-base-uncased style configuration
            bertconfig = AutoConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.bertmodel = AutoModel.from_pretrained('bert-base-uncased', config=bertconfig)
        elif self.config.text_encoder == 'roberta':
            bertconfig = AutoConfig.from_pretrained('roberta-large', output_hidden_states=True)
            self.bertmodel = AutoModel.from_pretrained('roberta-large', config=bertconfig)
        elif self.config.text_encoder == 'deberta':
            bertconfig = AutoConfig.from_pretrained('microsoft/deberta-v3-large', output_hidden_states=True)
            self.bertmodel = AutoModel.from_pretrained('microsoft/deberta-v3-large', config=bertconfig)
        self.layer_norm = nn.LayerNorm((self.hidden_size*2,))

        # defining the classifier, 3 layer MLP with dropout        
        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier_layer_1_dropout', nn.Dropout(self.dropout_rate))
        if self.config.text_encoder == 'glove':
            self.classifier.add_module('classifier_layer_1', nn.Linear(in_features=self.hidden_size*4, out_features=self.config.hidden_size*3))
        else:
            if self.config.text_encoder == 'bert':
                self.classifier.add_module('classifier_layer_1', nn.Linear(in_features=768, out_features=config.hidden_size*3))
            elif self.config.text_encoder == 'roberta':
                self.classifier.add_module('classifier_layer_1', nn.Linear(in_features=1024, out_features=config.hidden_size*3))
            elif self.config.text_encoder == 'deberta':
                self.classifier.add_module('classifier_layer_1', nn.Linear(in_features=1024, out_features=config.hidden_size*3))
        self.classifier.add_module('classifier_layer_1_activation', self.activation)

        self.classifier.add_module('classifier_layer_2_dropout', nn.Dropout(self.dropout_rate))
        self.classifier.add_module('classifier_layer_2', nn.Linear(in_features=self.config.hidden_size*3, out_features=self.config.hidden_size*2))
        self.classifier.add_module('classifier_layer_2_activation', self.activation)

        self.classifier.add_module('classifier_layer_3_dropout', nn.Dropout(self.dropout_rate))
        self.classifier.add_module('classifier_layer_3', nn.Linear(in_features=self.config.hidden_size*2, out_features=self.config.hidden_size))
        self.classifier.add_module('classifier_layer_3_activation', self.activation)
        
        self.classifier.add_module('classifier_layer_4_dropout', nn.Dropout(self.dropout_rate))
        self.classifier.add_module('classifier_layer_4', nn.Linear(in_features=self.config.hidden_size, out_features=self.output_size))


    def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask, hubert_feats=None, hubert_feats_att_mask=None, hubert_embeddings=None):
        batch_size = lengths.size(0)

        if self.config.text_encoder == 'glove':
            sentences = self.embed(sentences)
            # sentences = torch.permute(sentences, (1, 0, 2)) # dataloader returns batch first
            packed_sequence = pack_padded_sequence(sentences, lengths.cpu(), batch_first=True)
            if self.config.rnncell == 'lstm':
                packed_h1, (final_h1, _) = self.rnn1(packed_sequence)
            else:
                packed_h1, final_h1 = self.rnn1(packed_sequence)

            padded_h1, _ = pad_packed_sequence(packed_h1, batch_first=True, total_length=sentences.shape[1])
            normed_h1 = self.layer_norm(padded_h1)
            packed_normed_h1 = pack_padded_sequence(normed_h1, lengths.cpu(), batch_first=True)

            if self.config.rnncell == 'lstm':
                _, (final_h2, _) = self.rnn2(packed_normed_h1)
            else:
                _, final_h2 = self.rnn2(packed_normed_h1)
            features = torch.cat((final_h1, final_h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)    
        else:
            if self.config.text_encoder == 'bert':
                bert_output = self.bertmodel(input_ids=bert_sent,
                                             attention_mask=bert_sent_mask,
                                             token_type_ids=bert_sent_type)
            elif self.config.text_encoder == 'roberta':
                bert_output = self.bertmodel(input_ids=bert_sent,
                                             attention_mask=bert_sent_mask)
            elif self.config.text_encoder == 'deberta':
                bert_output = self.bertmodel(input_ids=bert_sent,
                                             attention_mask=bert_sent_mask)

            bert_output = bert_output[0]
            # masked mean
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
            features = bert_output

        # regression/classification
        out = self.classifier(features)
        return out, features


class VisualClassifier(nn.Module):
    def __init__(self, config):
        super(VisualClassifier, self).__init__()

        self.config = config
        self.input_size = config.visual_size
        self.hidden_size = int(config.visual_size)
        self.output_size = config.num_classes
        self.dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()

        rnn = nn.LSTM if self.config.rnncell == 'lstm' else nn.GRU

        # defining modules - two layer bidirectional LSTM with layer norm in between
        self.rnn1 = rnn(self.input_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.rnn2 = rnn(2*self.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.layer_norm = nn.LayerNorm((self.hidden_size*2,))
        
        # defining the classifier, 3 layer MLP with dropout        
        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier_layer_1_dropout', nn.Dropout(self.dropout_rate))
        self.classifier.add_module('classifier_layer_1', nn.Linear(in_features=self.hidden_size*4, out_features=self.config.hidden_size*3))
        self.classifier.add_module('classifier_layer_1_activation', self.activation)

        self.classifier.add_module('classifier_layer_2_dropout', nn.Dropout(self.dropout_rate))
        self.classifier.add_module('classifier_layer_2', nn.Linear(in_features=self.config.hidden_size*3, out_features= self.config.hidden_size*2))
        self.classifier.add_module('classifier_layer_2_activation', self.activation)

        self.classifier.add_module('classifier_layer_3_dropout', nn.Dropout(self.dropout_rate))
        self.classifier.add_module('classifier_layer_3', nn.Linear(in_features=self.config.hidden_size*2, out_features= self.config.hidden_size))
        self.classifier.add_module('classifier_layer_3_activation', self.activation)
        
        self.classifier.add_module('classifier_layer_4_dropout', nn.Dropout(self.dropout_rate))
        self.classifier.add_module('classifier_layer_4', nn.Linear(in_features=self.config.hidden_size, out_features= self.output_size))


    def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask, hubert_feats=None, hubert_feats_att_mask=None, hubert_embeddings=None):
        batch_size = lengths.size(0)
        # video = torch.permute(video, (1, 0, 2)) # dataloader returns batch first
        packed_sequence = pack_padded_sequence(video, lengths.cpu(), batch_first=True)

        if self.config.rnncell == 'lstm':
            packed_h1, (final_h1, _) = self.rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = self.rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1, batch_first=True, total_length=video.shape[1])
        normed_h1 = self.layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths.cpu(), batch_first=True)

        if self.config.rnncell == 'lstm':
            _, (final_h2, _) = self.rnn2(packed_normed_h1)
        else:
            _, final_h2 = self.rnn2(packed_normed_h1)
        features = torch.cat((final_h1, final_h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)    
        
        # regression/classification
        out = self.classifier(features)
        return out, features


class AcousticClassifier(nn.Module):
    def __init__(self, config):
        super(AcousticClassifier, self).__init__()

        self.config = config
        self.input_size = config.acoustic_size
        self.hidden_size = int(config.acoustic_size)
        self.output_size = config.num_classes
        self.dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()

        rnn = nn.LSTM if self.config.rnncell == 'lstm' else nn.GRU

        # defining modules - two layer bidirectional LSTM with layer norm in between
        self.rnn1 = rnn(self.input_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.rnn2 = rnn(2*self.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.layer_norm = nn.LayerNorm((self.hidden_size*2,))
        
        # defining the classifier, 3 layer MLP with dropout        
        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier_layer_1_dropout', nn.Dropout(self.dropout_rate))
        if self.config.audio_encoder == 'hubert':
            self.classifier.add_module('classifier_layer_1', nn.Linear(in_features=768, out_features=self.config.hidden_size*3))
        else:
            self.classifier.add_module('classifier_layer_1', nn.Linear(in_features=self.hidden_size*4, out_features=self.config.hidden_size*3))
        self.classifier.add_module('classifier_layer_1_activation', self.activation)

        self.classifier.add_module('classifier_layer_2_dropout', nn.Dropout(self.dropout_rate))
        self.classifier.add_module('classifier_layer_2', nn.Linear(in_features=self.config.hidden_size*3, out_features= self.config.hidden_size*2))
        self.classifier.add_module('classifier_layer_2_activation', self.activation)

        self.classifier.add_module('classifier_layer_3_dropout', nn.Dropout(self.dropout_rate))
        self.classifier.add_module('classifier_layer_3', nn.Linear(in_features=self.config.hidden_size*2, out_features= self.config.hidden_size))
        self.classifier.add_module('classifier_layer_3_activation', self.activation)
        
        self.classifier.add_module('classifier_layer_4_dropout', nn.Dropout(self.dropout_rate))
        self.classifier.add_module('classifier_layer_4', nn.Linear(in_features=self.config.hidden_size, out_features= self.output_size))


    def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask, hubert_feats=None, hubert_feats_att_mask=None, hubert_embeddings=None):
        if self.config.audio_encoder == 'hubert':
            features = hubert_embeddings
        else:
            batch_size = lengths.size(0)
            # acoustic = torch.permute(acoustic, (1, 0, 2)) # dataloader returns batch first
            packed_sequence = pack_padded_sequence(acoustic, lengths.cpu(), batch_first=True)

            if self.config.rnncell == 'lstm':
                packed_h1, (final_h1, _) = self.rnn1(packed_sequence)
            else:
                packed_h1, final_h1 = self.rnn1(packed_sequence)

            padded_h1, _ = pad_packed_sequence(packed_h1, batch_first=True, total_length=acoustic.shape[1])
            normed_h1 = self.layer_norm(padded_h1)
            packed_normed_h1 = pack_padded_sequence(normed_h1, lengths.cpu(), batch_first=True)

            if self.config.rnncell == 'lstm':
                _, (final_h2, _) = self.rnn2(packed_normed_h1)
            else:
                _, final_h2 = self.rnn2(packed_normed_h1)

            features = torch.cat((final_h1, final_h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)   
        # regression/classification
        out = self.classifier(features)
        return out, features



class EarlyFusion(nn.Module):
    def __init__(self, config):
        super(EarlyFusion, self).__init__()

        self.config = config
        
        self.text_size = config.embedding_size
        self.visual_size = config.visual_size
        self.acoustic_size = config.acoustic_size
        self.output_size = config.num_classes

        self.text_classifier = TextClassifier(config)
        self.visual_classifier = VisualClassifier(config)
        self.acoustic_classifier = AcousticClassifier(config)
        self.hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.dropout_rate = config.dropout
        self.activation = self.config.activation()
        
        # defining the classifier, 3 layer MLP with dropout        
        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier_layer_1_dropout', nn.Dropout(self.dropout_rate))
        if self.config.text_encoder == 'glove':
            if self.config.audio_encoder == 'hubert':
                self.classifier.add_module('classifier_layer_1', nn.Linear(in_features=sum(self.hidden_sizes[:2])*4+768, out_features=self.config.hidden_size*3))
            else:
                self.classifier.add_module('classifier_layer_1', nn.Linear(in_features=sum(self.hidden_sizes)*4, out_features=self.config.hidden_size*3))
        else:
            if self.config.text_encoder == 'bert' and self.config.audio_encoder != 'hubert':
                self.classifier.add_module('classifier_layer_1', nn.Linear(in_features=768+sum(self.hidden_sizes[1:])*4, out_features=config.hidden_size*3))
            elif self.config.text_encoder == 'roberta' and self.config.audio_encoder != 'hubert':
                self.classifier.add_module('classifier_layer_1', nn.Linear(in_features=1024+sum(self.hidden_sizes[1:])*4, out_features=config.hidden_size*3))
            elif self.config.text_encoder == 'deberta' and self.config.audio_encoder != 'hubert':
                self.classifier.add_module('classifier_layer_1', nn.Linear(in_features=1024+sum(self.hidden_sizes[1:])*4, out_features=config.hidden_size*3))
            elif self.config.text_encoder == 'bert' and self.config.audio_encoder == 'hubert':
                self.classifier.add_module('classifier_layer_1', nn.Linear(in_features=768*2+self.hidden_sizes[1]*4, out_features=config.hidden_size*3))
            elif self.config.text_encoder == 'roberta' and self.config.audio_encoder == 'hubert':
                self.classifier.add_module('classifier_layer_1', nn.Linear(in_features=1024+768+self.hidden_sizes[1]*4, out_features=config.hidden_size*3))
            elif self.config.text_encoder == 'deberta' and self.config.audio_encoder == 'hubert':
                self.classifier.add_module('classifier_layer_1', nn.Linear(in_features=1024+768+self.hidden_sizes[1]*4, out_features=config.hidden_size*3))
        self.classifier.add_module('classifier_layer_1_activation', self.activation)

        self.classifier.add_module('classifier_layer_2_dropout', nn.Dropout(self.dropout_rate))
        self.classifier.add_module('classifier_layer_2', nn.Linear(in_features=self.config.hidden_size*3, out_features= self.config.hidden_size*2))
        self.classifier.add_module('classifier_layer_2_activation', self.activation)

        self.classifier.add_module('classifier_layer_3_dropout', nn.Dropout(self.dropout_rate))
        self.classifier.add_module('classifier_layer_3', nn.Linear(in_features=self.config.hidden_size*2, out_features= self.config.hidden_size))
        self.classifier.add_module('classifier_layer_3_activation', self.activation)
        
        self.classifier.add_module('classifier_layer_4_dropout', nn.Dropout(self.dropout_rate))
        self.classifier.add_module('classifier_layer_4', nn.Linear(in_features=self.config.hidden_size, out_features= self.output_size))

    
    def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask, hubert_feats=None, hubert_feats_att_mask=None, hubert_embeddings=None):
        _, text_features = self.text_classifier(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask, hubert_feats, hubert_feats_att_mask, hubert_embeddings)
        _, visual_features = self.visual_classifier(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask, hubert_feats, hubert_feats_att_mask, hubert_embeddings)
        _, acoustic_features = self.acoustic_classifier(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask, hubert_feats, hubert_feats_att_mask, hubert_embeddings)

        features = torch.cat([text_features, visual_features, acoustic_features], dim=-1)

        # regression/classification
        out = self.classifier(features)
        return out, features

    
class LateFusion(nn.Module):
    def __init__(self, config, text_classifier=None, visual_classifier=None, acoustic_classifier=None):
        super(LateFusion, self).__init__()

        self.config = config
        if text_classifier is None:
            self.text_classifier = TextClassifier(config)
        else:
            self.text_classifier = text_classifier
        
        if visual_classifier is None:
            self.visual_classifier = VisualClassifier(config)
        else:
            self.visual_classifier = visual_classifier

        if acoustic_classifier is None:
            self.acoustic_classifier = AcousticClassifier(config)
        else:
            self.acoustic_classifier = acoustic_classifier
        

    def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask, hubert_feats=None, hubert_feats_att_mask=None, hubert_embeddings=None):
        text_outs, _ = self.text_classifier(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask, hubert_feats, hubert_feats_att_mask, hubert_embeddings)
        visual_outs, _  = self.visual_classifier(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask, hubert_feats, hubert_feats_att_mask, hubert_embeddings)
        acoustic_outs, _ = self.acoustic_classifier(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask, hubert_feats, hubert_feats_att_mask, hubert_embeddings)

        out = (text_outs + visual_outs + acoustic_outs) / 3

        return out, None