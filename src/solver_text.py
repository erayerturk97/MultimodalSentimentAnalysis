import sys
import time
import copy
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

from utils.eval_metrics import *
from utils.tools import *

from transformers import AutoModelForSequenceClassification

class Solver_Text(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True):
        self.hp = hp = hyp_params
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        if self.hp.bert_model == 'bert':
            self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
        elif self.hp.bert_model == 'roberta':
            self.model = AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=1)
        elif self.hp.bert_model == 'deberta':
            self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-large", num_labels=1)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.cuda()
        else:
            self.device = torch.device("cpu")

        # criterion
        self.criterion = criterion = nn.MSELoss(reduction="mean")

        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': hp.weight_decay_bert,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=hp.lr_bert)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=hp.when, factor=0.5, verbose=True)

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        model = self.model
        optimizer = self.optimizer

        # criterion for downstream task
        criterion = self.criterion

        def train(model, optimizer, criterion, stage=1):
            epoch_loss = 0

            model.train()
            num_batches = self.hp.n_train // self.hp.batch_size
            proc_loss, proc_size = 0, 0
            start_time = time.time()

            for i_batch, batch_data in enumerate(self.train_loader):
                model.zero_grad()
                text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data

                with torch.cuda.device(0):
                    text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
                    text.cuda(), visual.cuda(), audio.cuda(), y.cuda(), l.cuda(), bert_sent.cuda(), \
                    bert_sent_type.cuda(), bert_sent_mask.cuda()

                batch_size = y.size(0)

                if self.hp.bert_model == 'bert':
                    logits = model(bert_sent, bert_sent_type, bert_sent_mask).logits
                else:
                    logits = model(bert_sent, bert_sent_mask).logits

                loss = criterion(logits, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
                optimizer.step()

                proc_loss += loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += loss.item() * batch_size

                if i_batch % self.hp.log_interval == 0 and i_batch > 0:
                    avg_loss = proc_loss / proc_size
                    elapsed_time = time.time() - start_time
                    print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                        format(epoch, i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval, avg_loss))
                    proc_loss, proc_size = 0, 0
                    start_time = time.time()

            return epoch_loss / self.hp.n_train

        def evaluate(model, criterion, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0
            total_l1_loss = 0.0
        
            results = []
            truths = []

            with torch.no_grad():
                for batch in loader:
                    text, vision, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids = batch

                    with torch.cuda.device(0):
                        text, audio, vision, y = text.cuda(), audio.cuda(), vision.cuda(), y.cuda()
                        lengths = lengths.cuda()
                        bert_sent, bert_sent_type, bert_sent_mask = bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda()

                    batch_size = lengths.size(0) # bert_sent in size (bs, seq_len, emb_size)

                    if self.hp.bert_model == 'bert':
                        logits = model(bert_sent, bert_sent_type, bert_sent_mask).logits
                    else:
                        logits = model(bert_sent, bert_sent_mask).logits

                    criterion = nn.L1Loss()

                    total_loss += criterion(logits, y).item() * batch_size

                    # Collect the results into ntest if test else self.hp.n_valid)
                    results.append(logits)
                    truths.append(y)

            avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)

            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths

        best_valid = 1e8
        best_mae = 1e8
        patience = self.hp.patience

        best_model = copy.deepcopy(model)
        for epoch in range(1, self.hp.num_epochs+1):
            start = time.time()

            self.epoch = epoch

            # minimize all losses left
            train_loss = train(model, optimizer, criterion, 1)

            val_loss, _, _ = evaluate(model, criterion, test=False)
            test_loss, results, truths = evaluate(model, criterion, test=True)

            end = time.time()
            duration = end-start
            self.scheduler.step(val_loss)    # Decay learning rate by validation loss

            # validation F1
            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
            print("-"*50)
            
            if val_loss < best_valid:
                # update best validation
                patience = self.hp.patience
                best_valid = val_loss
                best_epoch = epoch

                if self.hp.dataset in ["mosei_senti", "mosei"]:
                    eval_mosei_senti(results, truths, True)
                elif self.hp.dataset == 'mosi':
                    eval_mosi(results, truths, True)

                best_results = results
                best_truths = truths
                best_model = copy.deepcopy(model)
                print(f"Saved model at pre_trained_models")
                save_model(self.hp, model)
            else:
                patience -= 1
                if patience == 0:
                    break

        print(f'Best epoch: {best_epoch}')
        if self.hp.dataset in ["mosei_senti", "mosei"]:
            self.best_dict = eval_mosei_senti(best_results, best_truths, True)
        elif self.hp.dataset == 'mosi':
            self.best_dict = eval_mosi(best_results, best_truths, True)
        sys.stdout.flush()
        self.best_dict["best_epoch"] = best_epoch
        return self.best_dict
