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
from model_fusion import MSA_EF, MSA_LF, USA_A, USA_V

class Solver_Fusion(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True):
        self.hp = hp = hyp_params
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.is_train = is_train

        # initialize the model
        if self.hp.fusion == 'early':
            self.model = model = MSA_EF(hp)
        elif self.hp.fusion == 'late':
            self.model = model = MSA_LF(hp)
        elif self.hp.fusion == 'audio':
            self.model = model = USA_A(hp)
        elif self.hp.fusion == 'video':
            self.model = model = USA_V(hp)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            model = model.cuda()
        else:
            self.device = torch.device("cpu")

        # criterion
        #self.criterion = criterion = nn.L1Loss(reduction="mean")
        self.criterion = criterion = nn.MSELoss(reduction="mean")

        # optimizer
        self.optimizer={}

        if self.is_train:
            main_param = []
            bert_param = []

            for name, p in model.named_parameters():
                if p.requires_grad:
                    if 'bert' in name:
                        bert_param.append(p)
                    else: 
                        main_param.append(p)

                for p in main_param:
                    if p.dim() > 1: # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                        nn.init.xavier_normal_(p)

        optimizer_main_group = [
            {'params': bert_param, 'weight_decay': hp.weight_decay_bert, 'lr': hp.lr_bert},
            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
        ]

        self.optimizer_main = getattr(torch.optim, self.hp.optim)(
            optimizer_main_group
        )
        self.scheduler_main = ReduceLROnPlateau(self.optimizer_main, mode='min', patience=hp.when, factor=0.5, verbose=True)

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        model = self.model
        optimizer_main = self.optimizer_main
        scheduler_main = self.scheduler_main

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

                text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
                    text.cuda(), visual.cuda(), audio.cuda(), y.cuda(), l.cuda(), bert_sent.cuda(), \
                    bert_sent_type.cuda(), bert_sent_mask.cuda()

                batch_size = y.size(0)

                preds = model(text, visual, audio, vlens, alens, 
                              bert_sent, bert_sent_type, bert_sent_mask)

                loss = criterion(preds, y)
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

                    preds = model(text, vision, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask)

                    if self.hp.dataset in ['mosi', 'mosei', 'mosei_senti'] and test:
                        criterion = nn.L1Loss()

                    total_loss += criterion(preds, y).item() * batch_size

                    # Collect the results into ntest if test else self.hp.n_valid)
                    results.append(preds)
                    truths.append(y)

            avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)

            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths

        best_valid = 1e8
        best_mae = 1e8
        best_model = copy.deepcopy(model)
        patience = self.hp.patience

        for epoch in range(1, self.hp.num_epochs+1):
            start = time.time()

            self.epoch = epoch

            # minimize all losses left
            train_loss = train(model, optimizer_main, criterion)

            val_loss, _, _ = evaluate(model, criterion, test=False)
            test_loss, results, truths = evaluate(model, criterion, test=True)

            end = time.time()
            duration = end-start
            scheduler_main.step(val_loss)    # Decay learning rate by validation loss

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
            eval_mosei_senti(best_results, best_truths, True)
        elif self.hp.dataset == 'mosi':
            self.best_dict = eval_mosi(best_results, best_truths, True)

        sys.stdout.flush()

    def test(self):
        model = self.model
        model = load_model(self.hp, model)
        model.eval()
        loader = self.test_loader
        results = []
        truths = []
        with torch.no_grad():
            for batch in loader:
                text, vision, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids = batch

                with torch.cuda.device(0):
                    text, audio, vision, y = text.cuda(), audio.cuda(), vision.cuda(), y.cuda()
                    lengths = lengths.cuda()
                    bert_sent, bert_sent_type, bert_sent_mask = bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda()

                if self.hp.test:
                    vision = torch.randn(vision.shape[0], vision.shape[1], vision.shape[2]).cuda()
                    audio = torch.randn(audio.shape[0], audio.shape[1], audio.shape[2]).cuda()

                preds = model(text, vision, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask)

                results.append(preds)
                truths.append(y)

        results = torch.cat(results)
        truths = torch.cat(truths)

        eval_mosei_senti(results, truths, True)
