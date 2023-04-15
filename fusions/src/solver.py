import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

import warnings
from utils import to_gpu, time_desc_decorator 
import models

class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True, model=None):

        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
    

    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):
        if self.model is None:
            self.model = getattr(models, self.train_config.model)(self.train_config)

        # Final list
        if self.train_config.model in ['TextClassifier', 'EarlyFusion', 'LateFusion']:
            for name, param in self.model.named_parameters():
                # Bert freezing customizations 
                if self.train_config.data == "mosei":
                    if "bertmodel.encoder.layer" in name:
                        layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                        if layer_num <= (8):
                            param.requires_grad = False
                elif self.train_config.data == "ur_funny":
                    if "bert" in name:
                        param.requires_grad = False

                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)

            # Initialize weight of Embedding matrix with Glove embeddings
            if self.train_config.text_encoder == 'glove':
                if self.train_config.model == 'TextClassifier':
                    if self.train_config.pretrained_emb is not None:
                        self.model.embed.weight.data = self.train_config.pretrained_emb
                    self.model.embed.requires_grad = False
                elif self.train_config.model in ['EarlyFusion', 'LateFusion']:
                    if self.train_config.pretrained_emb is not None:
                        self.model.text_classifier.embed.weight.data = self.train_config.pretrained_emb
                    self.model.text_classifier.embed.requires_grad = False


        if torch.cuda.is_available() and cuda:
            self.model.cuda()
            
        if self.is_train:    
        #     main_param = []
        #     bert_param = []

        #     for name, p in self.model.named_parameters():
        #         if p.requires_grad:
        #             if 'bert' in name:
        #                 bert_param.append(p)
        #             else: 
        #                 main_param.append(p)

        #         for p in main_param:
        #             if p.dim() > 1: # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
        #                 nn.init.xavier_normal_(p)
        #     optimizer_main_group = [
        #     {'params': bert_param, 'weight_decay': self.train_config.weight_decay_bert, 'lr': self.train_config.lr_bert},
        #     {'params': main_param, 'weight_decay': self.train_config.weight_decay_main, 'lr': self.train_config.lr_main}
        # ]

        #     self.optimizer = getattr(torch.optim, self.train_config.optimizer)(
        #         optimizer_main_group
            # )        
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)


    @time_desc_decorator('Training Start!')
    def train(self, cuda=True):

        if self.is_train:
            curr_patience = patience = self.train_config.patience
            num_trials = 1

            # self.criterion = criterion = nn.L1Loss(reduction="mean")
            if self.train_config.data == "ur_funny":
                self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
            else: # mosi and mosei are regression datasets
                self.criterion = criterion = nn.MSELoss(reduction="mean")

            
            best_valid_loss = float('inf')
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
            
            train_losses = []
            valid_losses = []
            for e in range(self.train_config.n_epoch):
                self.model.train()

                train_loss = []
                for batch in self.train_data_loader:
                    self.model.zero_grad()
                    t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch

                    batch_size = t.size(0)
                    t = to_gpu(t, on_cpu=not cuda)
                    v = to_gpu(v, on_cpu=not cuda)
                    a = to_gpu(a, on_cpu=not cuda)
                    y = to_gpu(y, on_cpu=not cuda)
                    l = to_gpu(l, on_cpu=not cuda)
                    bert_sent = to_gpu(bert_sent, on_cpu=not cuda)
                    bert_sent_type = to_gpu(bert_sent_type, on_cpu=not cuda)
                    bert_sent_mask = to_gpu(bert_sent_mask, on_cpu=not cuda)

                    y_tilde, _ = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)
                    
                    if self.train_config.data == "ur_funny":
                        y = y.squeeze()

                    loss = criterion(y_tilde, y)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.train_config.clip)
                    self.optimizer.step()
                    train_loss.append(loss.item())

                train_losses.append(train_loss)
                print(f"Training loss: {round(np.mean(train_loss), 4)}")

                valid_loss, valid_acc = self.eval(mode="dev")
                print(f"Val loss: {round(valid_loss, 4)}")
                
                print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
                if valid_loss <= best_valid_loss:
                    best_valid_loss = valid_loss
                    print("Found new best model on dev set!")
                    if not os.path.exists(f'{self.train_config.save_dir}/checkpoints/{self.train_config.model}'): os.makedirs(f'{self.train_config.save_dir}/checkpoints/{self.train_config.model}/')
                    torch.save(self.model.state_dict(), f'{self.train_config.save_dir}/checkpoints/{self.train_config.model}/model_{self.train_config.name}.std')
                    torch.save(self.optimizer.state_dict(), f'{self.train_config.save_dir}/checkpoints/{self.train_config.model}/optim_{self.train_config.name}.std')
                    curr_patience = patience
                else:
                    curr_patience -= 1
                    if curr_patience <= -1:
                        print("Running out of patience, loading previous best model.")
                        num_trials -= 1
                        curr_patience = patience
                        self.model.load_state_dict(torch.load(f'{self.train_config.save_dir}/checkpoints/{self.train_config.model}/model_{self.train_config.name}.std'))
                        self.optimizer.load_state_dict(torch.load(f'{self.train_config.save_dir}/checkpoints/{self.train_config.model}/optim_{self.train_config.name}.std'))
                        lr_scheduler.step()
                        print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
                
                if num_trials <= 0:
                    print("Running out of patience, early stopping.")
                    break

        self.eval(mode="test", to_print=True, cuda=cuda)

    
    def eval(self,mode=None, to_print=False, cuda=True):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []
        eval_loss = []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

            if to_print:
                self.model.load_state_dict(torch.load(
                                                    f'{self.train_config.save_dir}/checkpoints/{self.train_config.model}/model_{self.train_config.name}.std'))
                
                                                        
        with torch.no_grad():
            for batch in dataloader:
                self.model.zero_grad()
                t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch

                t = to_gpu(t, on_cpu=not cuda)
                v = to_gpu(v, on_cpu=not cuda)
                a = to_gpu(a, on_cpu=not cuda)
                y = to_gpu(y, on_cpu=not cuda)
                l = to_gpu(l, on_cpu=not cuda)
                bert_sent = to_gpu(bert_sent, on_cpu=not cuda)
                bert_sent_type = to_gpu(bert_sent_type, on_cpu=not cuda)
                bert_sent_mask = to_gpu(bert_sent_mask, on_cpu=not cuda)

                y_tilde, _ = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)

                if self.train_config.data == "ur_funny":
                    y = y.squeeze()
                
                criterion = nn.L1Loss()
                loss = criterion(y_tilde, y)

                eval_loss.append(loss.item())
                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        eval_loss = np.mean(eval_loss)
        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()

        accuracy = self.calc_metrics(y_true, y_pred, mode, to_print)
        return eval_loss, accuracy

    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
        """
        Metric scheme adapted from:
        https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
        """


        if self.train_config.data == "ur_funny":
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
            mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)
            mult_a5 = self.multiclass_acc(test_preds_a5, test_truth_a5)
            
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






