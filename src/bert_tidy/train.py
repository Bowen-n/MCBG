# created by Wu Bolun
# 2020.11.15
# bowenwu@sjtu.edu.cn

import os
import json
import tqdm
import argparse

import torch
import torch.nn as nn
from transformers import BertConfig

from vocab import WordVocab
from dataset import BERTDataset
from model import PREBERT, MLMBERT


class BaseTrainer(object):
    ''' Base class for training '''
    def __init__(self, model, train_loader, val_loader, 
                 optim, scheduler=None, criterion=None, name=None, multi_gpu=False):
        ''' criterion is optional '''
        self.name = name if name else 'model'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.train_loader = train_loader; self.train_size = len(train_loader.dataset)
        self.val_loader = val_loader; self.val_size = len(val_loader.dataset)

        self.optimizer = optim
        self.scheduler = scheduler
        self.criterion = criterion

        self.recorder = []
        self.multi_gpu = multi_gpu
    
    def print_record(self, train_loss, train_acc, val_loss, val_acc):
        ''' Print info after an epoch '''
        print('Epoch: {}/{}'.format(self.epoch, self.num_epochs-1))
        print('Train: loss {:.4f}, acc {:.4f}'.format(train_loss, train_acc))
        print('Val: loss {:.4f}, acc {:.4f}'.format(val_loss, val_acc))

        record = {'epoch': self.epoch, 'loss': val_loss, 'acc': val_acc}
        self.recorder.append(record)

    def save_model(self, save_dir):
        ''' save model '''
        save_path = os.path.join(save_dir, '{}_{}.pth'.format(self.name, self.epoch))
        if self.multi_gpu:
            torch.save(self.model.module.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)


    def train(self, num_epochs, savemodel_path='record', recorder_path='recorder.json'):
        ''' The whole training process '''
        self.num_epochs = num_epochs
        self.step = 0

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            
            # acc can be a `list` containing many accs
            train_loss, train_acc = self._train()
            val_loss, val_acc = self._val()

            self.print_record(train_loss, train_acc, val_loss, val_acc)
            self.save_model(savemodel_path)
        
        print(self.recorder)
        with open(recorder_path, 'w') as f:
            json.dump(self.recorder, f, indent=1)

    def _train(self):
        raise NotImplementedError

    def _val(self):
        raise NotImplementedError    


class PREBERTTrainer(BaseTrainer):
    ''' Train scheduler for PREBERT '''
    def __init__(self, bert: PREBERT, train_loader, val_loader,
                 optim, scheduler=None, criterion=None, name='prebert', multi_gpu=False):
        super(PREBERTTrainer, self).__init__(
            bert, train_loader, val_loader, optim, scheduler, criterion, name, multi_gpu)
    
    def print_record(self, train_loss, train_acc, val_loss, val_acc):
        ''' Print info after an epoch '''
        print('Epoch: {}/{}'.format(self.epoch, self.num_epochs-1))
        print('Train: loss {:.4f}, acc_nsp {:.4f}, acc_mlm {:.4f}'.format(
            train_loss, train_acc[0], train_acc[1]))
        print('Val: loss {:.4f}, acc_nsp {:.4f}, acc_mlm {:.4f}'.format(
            val_loss, val_acc[0], val_acc[1]))

        # records
        record = {'epoch': self.epoch, 'loss': val_loss,
                'acc_nsp': val_acc[0], 'acc_mlm': val_acc[1]}

        self.recorder.append(record)

    def _train(self):
        ''' Train for one epoch
        Ret: train_loss and train_acc for this epoch
             train_acc can be a `list` containing multiple accs
        '''
        self.model.train()

        epoch_loss = 0.0
        epoch_nsp_corr = 0
        epoch_mlm_corr = 0
        total_no_mask = 0

        iter_count = 0
        for data in train_loader:
            data = {key: value.to(self.device) for key, value in data.items()}

            loss, nsp_out, mlm_out = self.model(
                data['bert_input'],
                data['attention_mask'],
                data['segment_label'],
                data['bert_label'],
                data['is_next'])
            
            self.optimizer.zero_grad()
            if self.multi_gpu:
                loss.sum().backward()
                loss = loss.mean()
            else:
                loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * data['bert_label'].size(0)
            # nsp correct
            nsp_corr = nsp_out.argmax(dim=-1).eq(data['is_next']).sum().item()
            epoch_nsp_corr += nsp_corr
            # mlm correct
            no_mask = data['bert_label'] != -100
            no_mask_count = no_mask.sum().item()
            total_no_mask += no_mask_count
            mlm_pred = mlm_out.argmax(dim=-1)
            mlm_corr = (mlm_pred.eq(data['bert_label']) & no_mask).sum().item()
            epoch_mlm_corr += mlm_corr

            print('Step {}, Epoch {}/{}, iter {}/{}, Loss {:.4f}, Acc_nsp {:.4f}, Acc_mlm {:.4f}'.format(
                self.step, self.epoch, self.num_epochs-1,
                iter_count, int(self.train_size/BATCH_SIZE),
                loss.item(),
                nsp_corr/float(data['bert_label'].size(0)),
                mlm_corr/float(no_mask_count)))
            
            iter_count += 1
            self.step += 1
        
        train_loss = epoch_loss/self.train_size
        train_acc = [epoch_nsp_corr/self.train_size, epoch_mlm_corr/total_no_mask]
        return train_loss, train_acc
    
    def _val(self):
        ''' Validation on val_dataset
        Ret: val_loss and val_acc
             val_acc can be a `list` containing multiple accs
        '''
        self.model.eval()

        epoch_loss = 0.0
        epoch_nsp_corr = 0
        epoch_mlm_corr = 0
        total_no_mask = 0

        for data in tqdm.tqdm(self.val_loader):
            data = {key: value.to(self.device) for key, value in data.items()}

            loss, nsp_out, mlm_out = self.model(
                data['bert_input'],
                data['attention_mask'],
                data['segment_label'],
                data['bert_label'],
                data['is_next'])

            if self.multi_gpu:
                loss = loss.mean()

            epoch_loss += loss.item() * data['bert_label'].size(0)
            # nsp correct
            nsp_corr = nsp_out.argmax(dim=-1).eq(data['is_next']).sum().item()
            epoch_nsp_corr += nsp_corr
            # mlm correct
            no_mask = data['bert_label'] != -100
            no_mask_count = no_mask.sum().item()
            total_no_mask += no_mask_count
            mlm_pred = mlm_out.argmax(dim=-1)
            mlm_corr = (mlm_pred.eq(data['bert_label']) & no_mask).sum().item()
            epoch_mlm_corr += mlm_corr
        
        val_loss = epoch_loss/self.val_size
        val_acc = [epoch_nsp_corr/self.val_size, epoch_mlm_corr/total_no_mask]
        return val_loss, val_acc


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_path', type=str)
    parser.add_argument('-v', '--val_path', type=str)
    parser.add_argument('-c', '--vocab_path', type=str)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-m', '--multi_gpu', type=int, default=0)

    args = parser.parse_args()
    train_path = args.train_path
    val_path = args.val_path
    vocab_path = args.vocab_path
    BATCH_SIZE = args.batch_size   
    multi_gpu = bool(args.multi_gpu)

    # dataset
    vocab = WordVocab.load_vocab(vocab_path)
    train_dataset = BERTDataset(train_path, vocab, seq_len=64, ignored_label=-100)
    val_dataset = BERTDataset(val_path, vocab, seq_len=64, ignored_label=-100)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=5)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=5)
    print('Dataset loaded. Train: {}, Val: {}.'.format(len(train_dataset), len(val_dataset)))

    # model
    config = BertConfig(hidden_size=504, vocab_size=len(vocab), num_attention_heads=8, num_hidden_layers=8, return_dict=True)
    bert = PREBERT(config)
    if multi_gpu:
        bert = nn.DataParallel(bert, device_ids=[0, 1])
    bert.load_state_dict(torch.load('record/504-8-8-set_0.5_pair_30/prebert_39.pth'))

    print('Model loaded: {}'.format(config))
    
    # optim and criterion
    optimizer = torch.optim.AdamW(bert.parameters(), lr=1e-4)
    # optimizer = torch.optim.AdamW(bert.parameters(), lr=5e-5)

    # trainer
    trainer = PREBERTTrainer(
        bert=bert, train_loader=train_loader, val_loader=val_loader, 
        optim=optimizer, name='prebert', multi_gpu=multi_gpu)
    trainer.train(num_epochs=30)
