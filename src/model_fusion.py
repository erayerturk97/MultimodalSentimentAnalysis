import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from modules.encoders import LanguageEmbeddingLayer, RNNEncoder, SubNet

class MSA_EF(nn.Module):
    def __init__(self, hp):
        # Base Encoders
        super().__init__()
        self.hp = hp
        hp.d_tout = hp.d_tin

        self.text_enc = LanguageEmbeddingLayer(hp)
        self.visual_enc = RNNEncoder(
            in_size = hp.d_vin,
            hidden_size = hp.d_vh,
            out_size = hp.d_vout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )
        self.acoustic_enc = RNNEncoder(
            in_size = hp.d_ain,
            hidden_size = hp.d_ah,
            out_size = hp.d_aout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )

        dim_sum = hp.d_aout + hp.d_vout + hp.d_tout

        # Trimodal Settings
        self.fusion_prj = SubNet(
            in_size = dim_sum,
            hidden_size = hp.d_prjh,
            n_class = hp.n_class,
            dropout = hp.dropout_prj
        )

    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """
        enc_word = self.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask) # (batch_size, seq_len, emb_size)
        text = enc_word[:,0,:] # (batch_size, emb_size)

        acoustic = self.acoustic_enc(acoustic, a_len)
        visual = self.visual_enc(visual, v_len)

        # Linear proj and pred
        fusion, preds = self.fusion_prj(torch.cat([text, acoustic, visual], dim=1))

        return preds

class MSA_LF(nn.Module):
    def __init__(self, hp):
        # Base Encoders
        super().__init__()
        self.hp = hp
        hp.d_tout = hp.d_tin

        self.text_enc = LanguageEmbeddingLayer(hp)
        self.visual_enc = RNNEncoder(
            in_size = hp.d_vin,
            hidden_size = hp.d_vh,
            out_size = hp.d_vout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )
        self.acoustic_enc = RNNEncoder(
            in_size = hp.d_ain,
            hidden_size = hp.d_ah,
            out_size = hp.d_aout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )

        self.text_clf = SubNet(
            in_size = hp.d_tout,
            hidden_size = hp.d_prjh,
            n_class = hp.n_class,
            dropout = hp.dropout_prj
        )
        self.visual_clf = SubNet(
            in_size = hp.d_vout,
            hidden_size = hp.d_prjh,
            n_class = hp.n_class,
            dropout = hp.dropout_prj
        )
        self.acoustic_clf = SubNet(
            in_size = hp.d_aout,
            hidden_size = hp.d_prjh,
            n_class = hp.n_class,
            dropout = hp.dropout_prj
        )

    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """
        enc_word = self.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask) # (batch_size, seq_len, emb_size)
        text = enc_word[:,0,:] # (batch_size, emb_size)

        acoustic = self.acoustic_enc(acoustic, a_len)
        visual = self.visual_enc(visual, v_len)

        _, preds_t = self.text_clf(text)
        _, preds_v = self.visual_clf(visual)
        _, preds_a = self.acoustic_clf(acoustic)

        return (preds_t+preds_v+preds_a)/3

class USA_A(nn.Module):
    def __init__(self, hp):
        # Base Encoders
        super().__init__()
        self.hp = hp

        self.acoustic_enc = RNNEncoder(
            in_size = hp.d_ain,
            hidden_size = hp.d_ah,
            out_size = hp.d_aout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )
        self.acoustic_clf = SubNet(
            in_size = hp.d_aout,
            hidden_size = hp.d_prjh,
            n_class = hp.n_class,
            dropout = hp.dropout_prj
        )

    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask):
        acoustic = self.acoustic_enc(acoustic, a_len)
        _, preds_a = self.acoustic_clf(acoustic)

        return preds_a

class USA_V(nn.Module):
    def __init__(self, hp):
        # Base Encoders
        super().__init__()
        self.hp = hp

        self.visual_enc = RNNEncoder(
            in_size = hp.d_vin,
            hidden_size = hp.d_vh,
            out_size = hp.d_vout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )
        self.visual_clf = SubNet(
            in_size = hp.d_vout,
            hidden_size = hp.d_prjh,
            n_class = hp.n_class,
            dropout = hp.dropout_prj
        )

    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask):
        visual = self.visual_enc(visual, v_len)
        _, preds_v = self.visual_clf(visual)

        return preds_v
