import os
import sys
import time
import torch
import pickle
import numpy as np
import torch.optim as optim

from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

from src import models
from src import ctc
from src.utils import *
from src.eval_metrics import *


####################################################################
#
# Construct the model and the CTC module (which may not be needed)
#
####################################################################

def get_CTC_module(hyp_params):
    a2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_a, out_seq_len=hyp_params.l_len)
    v2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_v, out_seq_len=hyp_params.l_len)
    return a2l_module, v2l_module

def initiate(hyp_params, train_loader, valid_loader, test_loader, do_train=True):
    model = getattr(models, hyp_params.model+'Model')(hyp_params)

    
    for name, param in model.named_parameters():
        # Bert freezing customizations 
        if hyp_params.dataset == "mosei":
            if "bertmodel.encoder.layer" in name:
                layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                if layer_num <= (8):
                    param.requires_grad = False
        elif hyp_params.dataset == "ur_funny":
            if "bert" in name:
                param.requires_grad = False

        if 'weight_hh' in name:
            nn.init.orthogonal_(param)
        #print('\t' + name, param.requires_grad)

    # Initialize weight of Embedding matrix with Glove embeddings
    if hyp_params.text_encoder == 'glove':
        if hyp_params.pretrained_emb is not None:
            model.embed.weight.data = hyp_params.pretrained_emb
        model.embed.requires_grad = False


    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = torch.nn.MSELoss(reduction='mean')

    if hyp_params.aligned or hyp_params.model=='MULT':
        ctc_criterion = None
        ctc_a2l_module, ctc_v2l_module = None, None
        ctc_a2l_optimizer, ctc_v2l_optimizer = None, None
    else:
        from warpctc_pytorch import CTCLoss
        ctc_criterion = CTCLoss()
        ctc_a2l_module, ctc_v2l_module = get_CTC_module(hyp_params)
        if hyp_params.use_cuda:
            ctc_a2l_module, ctc_v2l_module = ctc_a2l_module.cuda(), ctc_v2l_module.cuda()
        ctc_a2l_optimizer = getattr(optim, hyp_params.optim)(ctc_a2l_module.parameters(), lr=hyp_params.lr)
        ctc_v2l_optimizer = getattr(optim, hyp_params.optim)(ctc_v2l_module.parameters(), lr=hyp_params.lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'ctc_a2l_module': ctc_a2l_module,
                'ctc_v2l_module': ctc_v2l_module,
                'ctc_a2l_optimizer': ctc_a2l_optimizer,
                'ctc_v2l_optimizer': ctc_v2l_optimizer,
                'ctc_criterion': ctc_criterion,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader, do_train=do_train)

####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader, do_train=True):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']    
    
    ctc_a2l_module = settings['ctc_a2l_module']
    ctc_v2l_module = settings['ctc_v2l_module']
    ctc_a2l_optimizer = settings['ctc_a2l_optimizer']
    ctc_v2l_optimizer = settings['ctc_v2l_optimizer']
    ctc_criterion = settings['ctc_criterion']

    scheduler = settings['scheduler']

    def train(model, optimizer, criterion, ctc_a2l_module, ctc_v2l_module, ctc_a2l_optimizer, ctc_v2l_optimizer, ctc_criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, batch in enumerate(train_loader):
            text, vision, audio, eval_attr, _, bert_sent, bert_sent_type, bert_sent_mask, _, _, _, hubert_embeddings_all = batch

            model.zero_grad()
            if ctc_criterion is not None:
                ctc_a2l_module.zero_grad()
                ctc_v2l_module.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr, bert_sent, bert_sent_type, bert_sent_mask, hubert_embeddings_all = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda(), bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda(), hubert_embeddings_all.cuda()
            
            batch_size = eval_attr.size(0)

            ######## CTC STARTS ######## Do not worry about this if not working on CTC
            if ctc_criterion is not None:
                ctc_a2l_net = ctc_a2l_module
                ctc_v2l_net = ctc_v2l_module

                audio, a2l_position = ctc_a2l_net(audio) # audio now is the aligned to text
                vision, v2l_position = ctc_v2l_net(vision)

                ## Compute the ctc loss
                l_len, a_len, v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
                # Output Labels
                l_position = torch.tensor([i+1 for i in range(l_len)]*batch_size).int().cpu()
                # Specifying each output length
                l_length = torch.tensor([l_len]*batch_size).int().cpu()
                # Specifying each input length
                a_length = torch.tensor([a_len]*batch_size).int().cpu()
                v_length = torch.tensor([v_len]*batch_size).int().cpu()

                ctc_a2l_loss = ctc_criterion(a2l_position.transpose(0,1).cpu(), l_position, a_length, l_length)
                ctc_v2l_loss = ctc_criterion(v2l_position.transpose(0,1).cpu(), l_position, v_length, l_length)
                ctc_loss = ctc_a2l_loss + ctc_v2l_loss
                ctc_loss = ctc_loss.cuda() if hyp_params.use_cuda else ctc_loss
            else:
                ctc_loss = 0
            ######## CTC ENDS ########

            combined_loss = 0
            preds, hiddens = model(text, audio, vision, bert_sent, bert_sent_type, bert_sent_mask, hubert_embeddings_all)
            raw_loss = criterion(preds, eval_attr)
            combined_loss = raw_loss + ctc_loss
            combined_loss.backward()

            if ctc_criterion is not None:
                torch.nn.utils.clip_grad_norm_(ctc_a2l_module.parameters(), hyp_params.clip)
                torch.nn.utils.clip_grad_norm_(ctc_v2l_module.parameters(), hyp_params.clip)
                ctc_a2l_optimizer.step()
                ctc_v2l_optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()

        return epoch_loss / hyp_params.n_train

    def evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=False, to_print=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for i_batch, batch in enumerate(loader):
                text, vision, audio, eval_attr, _, bert_sent, bert_sent_type, bert_sent_mask, _, _, _, hubert_embeddings_all = batch

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr, bert_sent, bert_sent_type, bert_sent_mask, hubert_embeddings_all  = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda(), bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda(), hubert_embeddings_all.cuda()

                batch_size = eval_attr.size(0)

                if (ctc_a2l_module is not None) and (ctc_v2l_module is not None):
                    ctc_a2l_net = ctc_a2l_module
                    ctc_v2l_net = ctc_v2l_module
                    audio, _ = ctc_a2l_net(audio)     # audio aligned to text
                    vision, _ = ctc_v2l_net(vision)   # vision aligned to text

                preds, _ = model(text, audio, vision, bert_sent, bert_sent_type, bert_sent_mask, hubert_embeddings_all)
                total_loss += criterion(preds, eval_attr).item() * batch_size

                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        
        accuracy = calc_metrics(hyp_params, truths, results, to_print=to_print)
        return avg_loss, results, truths

    if do_train:
        best_valid = 1e8
        for epoch in range(1, hyp_params.num_epochs+1):
            start = time.time()
            train(model, optimizer, criterion, ctc_a2l_module, ctc_v2l_module, ctc_a2l_optimizer, ctc_v2l_optimizer, ctc_criterion)
            val_loss, _, _ = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=False)
            test_loss, _, _ = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=True)
            
            end = time.time()
            duration = end-start
            scheduler.step(val_loss)    # Decay learning rate by validation loss

            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
            print("-"*50)
            
            if val_loss < best_valid:
                print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
                save_model(hyp_params, model, name=hyp_params.name)
                best_valid = val_loss

    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=True, to_print=True)

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True)

    sys.stdout.flush()



def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def calc_metrics(hyp_params, y_true, y_pred, mode=None, to_print=False):
    """
    Metric scheme adapted from:
    https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
    """
    if isinstance(y_true, torch.Tensor): 
        y_true = np.array(y_true.cpu()).reshape(-1)
    if isinstance(y_pred, torch.Tensor):      
        y_pred = np.array(y_pred.cpu()).reshape(-1)

    if hyp_params.dataset == "ur_funny":
        test_preds = np.argmax(y_pred, 1)
        test_truth = y_true

        if to_print:
            print("Confusion Matrix (pos/neg) :")
            print(confusion_matrix(test_truth, test_preds))
            print("Classification Report (pos/neg) :")
            print(classification_report(test_truth, test_preds, digits=5))
            print("Accuracy (pos/neg) ", accuracy_score(test_truth, test_preds))
        
        return accuracy_score(test_truth, test_preds)

    else:
        test_preds = y_pred
        test_truth = y_true

        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

        mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
        
        f_score = f1_score((test_truth[non_zeros] > 0), (test_preds[non_zeros] > 0), average='weighted')
        
        # pos - neg
        binary_truth = (test_truth[non_zeros] > 0)
        binary_preds = (test_preds[non_zeros] > 0)

        if to_print:
            print("mae: ", mae)
            print("corr: ", corr)
            print("mult_acc: ", mult_a7)
            print("Classification Report (pos/neg) :")
            print(classification_report(binary_truth, binary_preds, digits=5))
            print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
        
        # non-neg - neg
        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)

        if to_print:
            print("Classification Report (non-neg/neg) :")
            print(classification_report(binary_truth, binary_preds, digits=5))
            print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
        
        return accuracy_score(binary_truth, binary_preds)