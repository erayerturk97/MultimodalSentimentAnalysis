import sys
import time
import copy
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

from utils.eval_metrics import *
from utils.tools import *
from model_gb import MSA_GB

class Solver_GB(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, use_cuda = True):
        self.hp = hp = hyp_params
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.use_cuda = use_cuda
        if torch.cuda.is_available() and self.use_cuda:
            generator = torch.Generator(device='cuda')
        else:
            generator = torch.Generator(device='cpu')

        v_rate = 0.1
        train_datas = train_loader.dataset
        splitloc = int(len(train_datas)*v_rate)
        inds = list(range(len(train_datas)))
        t_inds = inds[splitloc:]
        v_inds = inds[:splitloc]
        tt_data = Subset(train_datas, t_inds)
        tv_data = Subset(train_datas, v_inds)

        self.tt_loader = DataLoader(
            dataset=tt_data,
            shuffle=True,
            batch_size=train_loader.batch_size,
            collate_fn=train_loader.collate_fn,
            generator=generator)
        self.tv_loader = DataLoader(
            dataset=tv_data,
            shuffle=False,
            batch_size=train_loader.batch_size,
            collate_fn=train_loader.collate_fn,
            generator=generator)

        self.is_train = is_train

        # initialize the model
        self.model = model = MSA_GB(hp)
        if torch.cuda.is_available() and self.use_cuda:
            self.device = torch.device("cuda")
            model = model.cuda()
        else:
            self.device = torch.device("cpu")

        # criterion
        self.criterion = criterion = nn.MSELoss(reduction="mean")

        # optimizer
        self.optimizer_main, self.scheduler_main = self.get_optim(self.model)

    def get_optim(self, model):
        # optimizer
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
            {'params': bert_param, 'weight_decay': self.hp.weight_decay_bert, 'lr': self.hp.lr_bert},
            {'params': main_param, 'weight_decay': self.hp.weight_decay_main, 'lr': self.hp.lr_main}
        ]
        optimizer_main = getattr(torch.optim, self.hp.optim)(
            optimizer_main_group
        )
        scheduler_main = ReduceLROnPlateau(optimizer_main, mode='min', patience=self.hp.when, factor=0.5, verbose=True)

        return optimizer_main, scheduler_main

    def gb_train(self, model, optimizer, idx):
        model.train()
        #ltN, _, _ = self.evaluate(model, self.criterion, loader=self.tt_loader, index=idx)
        #ltN, _, _ = self.evaluate(model, self.criterion, loader=self.tv_loader, index=idx)
        for epoch in range(self.hp.num_gb_epochs):
            for i_batch, batch_data in enumerate(self.tt_loader):
                model.zero_grad()
                text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data
                if torch.cuda.is_available() and self.use_cuda:
                    with torch.cuda.device(0):
                        text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
                        text.cuda(), visual.cuda(), audio.cuda(), y.cuda(), l.cuda(), bert_sent.cuda(), \
                        bert_sent_type.cuda(), bert_sent_mask.cuda()

                preds = model(text, visual, audio, vlens, alens, 
                            bert_sent, bert_sent_type, bert_sent_mask)
                loss = self.criterion(preds[idx], y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
                optimizer.step()

        ltNn, _, _ = self.evaluate(model, self.criterion, loader=self.tt_loader, index=idx)
        lvNn, _, _ = self.evaluate(model, self.criterion, loader=self.tv_loader, index=idx)

        oNn = lvNn-ltNn
        if oNn < 0:
            oNn = 0.0001

        return abs(lvNn/(oNn**2))

    def gb_estimate(self, model):
        weights = []
        for modal_idx in range(3):
            print("At gb_estimate unimodal "+str(modal_idx))
            uni_model = copy.deepcopy(model)
            if torch.cuda.is_available() and self.use_cuda:
                uni_model = uni_model.cuda()
            uni_optim, _ = self.get_optim(uni_model)
            w = self.gb_train(uni_model, uni_optim, modal_idx)
            weights.append(w)

        print("At gb_estimate multimodal ")
        tri_model = copy.deepcopy(model)
        if torch.cuda.is_available() and self.use_cuda:
                tri_model = tri_model.cuda()
        tri_optim, _ = self.get_optim(tri_model)
        w = self.gb_train(uni_model, uni_optim, 3)
        weights.append(w)

        return weights/np.sum(np.array(weights))

    def train(self, model, optimizer, criterion, weights, idx=-1):
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
            if idx == -1:
                loss_t = criterion(preds[0], y)
                loss_v = criterion(preds[1], y)
                loss_a = criterion(preds[2], y)
                loss_tri = criterion(preds[3], y)
                loss = loss_t*weights[0]+loss_v*weights[1]+loss_a*weights[2]+loss_tri*weights[3]
            else:
                preds = preds[idx]
                loss = criterion(preds, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
            optimizer.step()

            proc_loss += loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += loss.item() * batch_size

        return epoch_loss / self.hp.n_train

    def evaluate(self, model, criterion, loader=None, test=False, index=0):
        model.eval()
        if loader == None:
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
                preds = preds[index]

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

    def train_and_eval(self):
        model = self.model
        optimizer_main = self.optimizer_main
        scheduler_main = self.scheduler_main

        # criterion for downstream task
        criterion = self.criterion

        best_valid = 1e8
        best_mae = 1e8
        best_model = copy.deepcopy(model)

        weights = self.gb_estimate(model)
        print("weights: " + str(weights))
        for epoch in range(1, self.hp.num_epochs+1):
            start = time.time()

            self.epoch = epoch

            # minimize all losses left
            train_loss = self.train(model, optimizer_main, criterion, weights)

            val_loss, _, _ = self.evaluate(model, criterion, test=False, index=0)
            test_loss, results, truths = self.evaluate(model, criterion, test=True, index=0)

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
            self.best_dict = eval_mosei_senti(best_results, best_truths, True)
        elif self.hp.dataset == 'mosi':
            self.best_dict = eval_mosi(best_results, best_truths, True)

        sys.stdout.flush()
        self.best_dict["best_epoch"] = best_epoch
        return self.best_dict

    def test(self, index=0):
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
                preds = preds[index]

                results.append(preds)
                truths.append(y)

        results = torch.cat(results)
        truths = torch.cat(truths)

        eval_mosei_senti(results, truths, True)
