# created by Wu Bolun
# 2020.11.13
# bowenwu@sjtu.edu.cn

import csv
import json
import os
import re
import sys

import numpy as np
import torch
import tqdm
from torch_geometric.data import Data, DataLoader, Dataset
from transformers import BertConfig

sys.path.append('..')
from bert_tidy.model import PREBERT
from bert_tidy.vocab import *

sys.path.append('../dataset/')
from bert_data import REGS, TYPES
from constant import *


class BBSDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, segment_labels, attention_mask):
        self.inputs = inputs # (total, seq_len)
        self.segment_labels = segment_labels # (total, seq_len)
        self.attention_mask = attention_mask
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_ = self.inputs[idx]
        segment_label_ = self.segment_labels[idx]
        attention_mask_ = self.attention_mask[idx]

        return {
            'inputs': torch.tensor(input_),
            'segment_labels': torch.tensor(segment_label_),
            'attention_mask': torch.tensor(attention_mask_)
        }


class CFGDataset(Dataset):
    def __init__(self, root):
        self.number_of_classes = 9
        self.label_path = os.path.join(root, 'trainLabels.csv')
        
        # labels
        self.labels = {}
        with open(self.label_path, 'r') as f:
            data = csv.reader(f)
            for row in data:
                if row[0] == 'Id':
                    continue
                self.labels[row[0]] = int(row[1])

        super(CFGDataset, self).__init__(root, transform=None, pre_transform=None)
    
    @property
    def num_classes(self):
        return self.number_of_classes
    
    @property
    def raw_dir(self):
        return os.path.join(self.root, 'cfg')
    
    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)
    
    @property
    def processed_file_names(self):
        file_names = ['data_{}_{}.pt'.format(
            i, filename.split('.')[0]) for i, filename in enumerate(self.raw_file_names)]
        return file_names
    
    def raw_file_name_lookup(self, idx):
        return self.raw_file_names[idx]
    
    def train_val_split(self, k):
        ''' k fold train_val_split 
        k = 0, 1, 2, 3, 4
        '''
        # TODO: 5-fold
        # 0 0.2 0.4 0.6 0.8
        label_list = list()
        for filename in self.raw_file_names:
            label = self.labels[filename.split('.')[0]]
            label_list.append(int(label))

        groups = [[] for _ in range(9)]
        for i, label in enumerate(label_list):
            groups[label-1].append(i)
        
        train_idx = []; val_idx = []

        for group in groups:
            group_len = len(group)
            slice_1 = int(group_len*0.2*k)
            slice_2 = int(group_len*0.2*(k+1))

            train_idx.extend(group[0:slice_1])
            train_idx.extend(group[slice_2:])

            val_idx.extend(group[slice_1:slice_2])
            # train_idx.extend(group[int(len(group) * 0.15):])
            # val_idx.extend(group[:int(len(group) * 0.15)])
        
        return train_idx, val_idx

    def len(self):
        return len(self.processed_file_names)

    def process(self):
        raise NotImplementedError
    
    def get(self, idx):
        raw_file_name = self.raw_file_name_lookup(idx).split('.')[0]
        data = torch.load(os.path.join(self.processed_dir, 'data_{}_{}.pt'.format(idx, raw_file_name)))
        return data


class CFGDataset_Opcode(CFGDataset):
    def __init__(self, root='/home/wubolun/data/malware/big2015',
                 vocab_path=None, seq_len=None):
        self.vocab_path = vocab_path if vocab_path else os.path.join(root, 'bbs_vocab.small')
        self.seq_len = seq_len if seq_len else 20

        super(CFGDataset_Opcode, self).__init__(root, transform=None, pre_transform=None)

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'cfg_processed_opcode')

    def process(self):
        ''' process raw JSON CFGs '''

        # vocab
        self.vocab = WordVocab.load_vocab(self.vocab_path)
        
        idx = 0
        for raw_path in tqdm.tqdm(self.raw_file_names):
            fullpath = os.path.join(self.raw_dir, raw_path)
            with open(fullpath, 'r') as f:
                cfg = json.load(f)
            
            ## y (label)
            y = int(self.labels[raw_path.split('.')[0]]) - 1
     
            addr_to_id = dict() # {str: int}
            current_node_id = -1

            x = list() # node attributes
            for addr, block in cfg.items(): # addr is 'str
                current_node_id += 1
                addr_to_id[addr] = current_node_id

                # get tokenized opcode sequence as node attributes
                tokenized_seq = []
                for insn in block['insn_list']:
                    opcode = insn['opcode']
                    tokenized = self.vocab.stoi.get(opcode, self.vocab.unk_index)
                    tokenized_seq.append(tokenized)
                # add [CLS] and [SEP]
                tokenized_seq = [self.vocab.sos_index] + tokenized_seq + [self.vocab.eos_index]
                # max seq len
                tokenized_seq = tokenized_seq[:self.seq_len]
                # padding
                padding = [self.vocab.pad_index for _ in range(self.seq_len - len(tokenized_seq))]
                tokenized_seq.extend(padding)
                
                x.append(tokenized_seq)
            
            # get sparse adjacent matrix
            edge_index = list()
            for addr, block in cfg.items(): # addr is `str`
                start_nid = addr_to_id[addr]
                for out_edge in block['out_edge_list']:
                    end_nid = addr_to_id[str(out_edge)]

                    ## edge_index
                    edge_index.append([start_nid, end_nid])
            
            # Data
            x = torch.tensor(x)
            y = torch.tensor(y, dtype=torch.long).unsqueeze(0)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            data = Data(x=x, edge_index=edge_index, y=y)

            # save
            assert(self.raw_file_name_lookup(idx) == raw_path)
            save_path = 'data_{}_{}.pt'.format(idx, raw_path.split('.')[0])
            save_path = os.path.join(self.processed_dir, save_path)
            torch.save(data, save_path)

            idx += 1


class CFGDataset_Normalized(CFGDataset):
    def __init__(self, root='/home/wubolun/data/malware/big2015',
                 vocab_path=None, seq_len=None):
        self.vocab_path = vocab_path if vocab_path else os.path.join(root, 'bbs_vocab_normal.small')
        self.seq_len = seq_len if seq_len else 20
            
        super(CFGDataset_Normalized, self).__init__(root)
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, 'cfg_processed_normal')
    
    def normalize(self, opcode_operands):
        ''' normalization '''
        if opcode_operands[0] == 'call':
            return 'call'

        ret_ins_str = opcode_operands[0]

        for operand in opcode_operands[1:]:

            if operand in REGS:
                ret_ins_str += '_{}'.format(operand)
            elif operand.startswith('[') and operand.endswith(']'):
                ret_ins_str += '_{}'.format(self._handle_ptr(operand))
            elif operand.startswith('ds:') or '_' in operand:
                ret_ins_str += '_MEM'
            elif operand.isnumeric() or operand.endswith('h'):
                ret_ins_str += '_IMM'
            elif operand in TYPES:
                continue
            else:
                ret_ins_str += '_{}'.format(operand)
        
        return ret_ins_str

    def _handle_ptr(self, ptr):
        ''' 
        [ebp-1Ch] [ebp+8+4] [esp+40h+-18h] [ebp+esp*4] [ebp+8]
        '''

        def _judge_num(string):
            try:
                if string.endswith('h'):
                    tmp = int(string[:-1], 16)
                    return True
                else:
                    return False
            except:
                return False

        # print(ptr, end='\t')
        ptr = ptr.replace('+-', '-')

        ret_ptr = '['
        item = ''
        count = 0
        operator = ''

        for char in ptr[1:]:
            if char in ['+', '-', ']']:
                if not item.isnumeric() and not _judge_num(item):
                    ret_ptr += operator + item
                else:
                    if item.isnumeric():
                        value = int(item)
                    else:
                        value = int('0x'+item[:-1], 16)

                    if operator == '+':
                        count += value
                    elif operator == '-':
                        count -= value
                operator = char if char != ']' else ''
                item = ''
            else:
                item += char
        
        # print(count, end='\t')
        if count <= -10:
            ret_ptr += '-' + (hex(count)[3:]).upper() + 'h]'
        elif -10 < count < 0:
            ret_ptr += '-' + (hex(count)[3:]).upper() + ']'
        elif count == 0:
            ret_ptr += ']'
        elif 0 < count < 10:
            ret_ptr += '+' + (hex(count)[2:]).upper() + ']'
        elif count >= 10:
            ret_ptr += '+' + (hex(count)[2:]).upper() + 'h]'
        
        # print(ret_ptr)
        return ret_ptr

    def process(self):
        ''' process raw JSON CFGs '''

        # vocab
        self.vocab = WordVocab.load_vocab(self.vocab_path)

        idx = 0
        for raw_path in tqdm.tqdm(self.raw_file_names):
            fullpath = os.path.join(self.raw_dir, raw_path)
            with open(fullpath, 'r') as f:
                cfg = json.load(f)
            
            ## y (label)
            y = int(self.labels[raw_path.split('.')[0]]) - 1
     
            addr_to_id = dict() # {str: int}
            current_node_id = -1

            x = list() # node attributes
            seg = list() # node segment labels
            for addr, block in cfg.items(): # addr is 'str
                current_node_id += 1
                addr_to_id[addr] = current_node_id

                # get tokenized opcode sequence as node attributes
                tokenized_seq = []
                for insn in block['insn_list']:
                    opcode = insn['opcode']
                    operands = insn['operands']

                    opcode_operands = [opcode] + operands
                    normalized = self.normalize(opcode_operands)

                    tokenized = self.vocab.stoi.get(normalized, self.vocab.unk_index)
                    tokenized_seq.append(tokenized)

                # add [CLS] and [SEP]
                tokenized_seq = [self.vocab.sos_index] + tokenized_seq + [self.vocab.eos_index]
                segment_label = [1 for _ in range(len(tokenized_seq))]
                # max seq len
                tokenized_seq = tokenized_seq[:self.seq_len]
                segment_label = segment_label[:self.seq_len]
                # padding
                padding = [self.vocab.pad_index for _ in range(self.seq_len - len(tokenized_seq))]
                tokenized_seq.extend(padding); segment_label.extend(padding)
                
                x.append(tokenized_seq)
                seg.append(segment_label)
            
            # get sparse adjacent matrix
            edge_index = list()
            for addr, block in cfg.items(): # addr is `str`
                start_nid = addr_to_id[addr]
                for out_edge in block['out_edge_list']:
                    end_nid = addr_to_id[str(out_edge)]

                    ## edge_index
                    edge_index.append([start_nid, end_nid])
            
            # Data
            x = torch.tensor(x)
            seg = torch.tensor(seg)
            y = torch.tensor(y, dtype=torch.long).unsqueeze(0)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            data = Data(x=x, seg=seg, edge_index=edge_index, y=y)

            # save
            assert(self.raw_file_name_lookup(idx) == raw_path)
            save_path = 'data_{}_{}.pt'.format(idx, raw_path.split('.')[0])
            save_path = os.path.join(self.processed_dir, save_path)
            torch.save(data, save_path)

            idx += 1


class CFGDataset_Normalized_After_BERT(CFGDataset):
    def __init__(self, root='/home/wubolun/data/malware/big2015',
                 vocab_path=None, seq_len=None):
        self.vocab_path = vocab_path if vocab_path else os.path.join(root, 'bbs_vocab_normal.small')
        self.seq_len = seq_len if seq_len else 64
        
        super(CFGDataset_Normalized_After_BERT, self).__init__(root)
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, 'cfg_after_prebert-504-8-8')
    
    def normalize(self, opcode_operands):
        ''' normalization '''
        if opcode_operands[0] == 'call':
            return 'call'

        ret_ins_str = opcode_operands[0]

        for operand in opcode_operands[1:]:

            if operand in REGS:
                ret_ins_str += '_{}'.format(operand)
            elif operand.startswith('[') and operand.endswith(']'):
                ret_ins_str += '_{}'.format(self._handle_ptr(operand))
            elif operand.startswith('ds:') or '_' in operand:
                ret_ins_str += '_MEM'
            elif operand.isnumeric() or operand.endswith('h'):
                ret_ins_str += '_IMM'
            elif operand in TYPES:
                continue
            else:
                ret_ins_str += '_{}'.format(operand)
        
        return ret_ins_str

    def _handle_ptr(self, ptr):
        ''' 
        [ebp-1Ch] [ebp+8+4] [esp+40h+-18h] [ebp+esp*4] [ebp+8]
        '''

        def _judge_num(string):
            try:
                if string.endswith('h'):
                    tmp = int(string[:-1], 16)
                    return True
                else:
                    return False
            except:
                return False

        # print(ptr, end='\t')
        ptr = ptr.replace('+-', '-')

        ret_ptr = '['
        item = ''
        count = 0
        operator = ''

        for char in ptr[1:]:
            if char in ['+', '-', ']']:
                if not item.isnumeric() and not _judge_num(item):
                    ret_ptr += operator + item
                else:
                    if item.isnumeric():
                        value = int(item)
                    else:
                        value = int('0x'+item[:-1], 16)

                    if operator == '+':
                        count += value
                    elif operator == '-':
                        count -= value
                operator = char if char != ']' else ''
                item = ''
            else:
                item += char
        
        # print(count, end='\t')
        if count <= -10:
            ret_ptr += '-' + (hex(count)[3:]).upper() + 'h]'
        elif -10 < count < 0:
            ret_ptr += '-' + (hex(count)[3:]).upper() + ']'
        elif count == 0:
            ret_ptr += ']'
        elif 0 < count < 10:
            ret_ptr += '+' + (hex(count)[2:]).upper() + ']'
        elif count >= 10:
            ret_ptr += '+' + (hex(count)[2:]).upper() + 'h]'
        
        # print(ret_ptr)
        return ret_ptr

    def process(self):
        ''' process raw JSON CFGs '''

        # vocab
        self.vocab = WordVocab.load_vocab(self.vocab_path)

        # `PREBERT`
        config = BertConfig(hidden_size=504, 
            vocab_size=len(self.vocab), num_attention_heads=8, 
            num_hidden_layers=8, return_dict=True)
        self.model = PREBERT(config) # PREBERT

        self.model.load_state_dict(torch.load('../bert_tidy/record/504-8-8-set_0.5_pair_30/prebert_39.pth'))
        print('Loaded BERT model.')
        # `transformers.BertModel`
        self.bert = self.model.bert.bert 
        self.bert = self.bert.cuda()
        self.bert.eval()

        idx = 0
        for raw_path in tqdm.tqdm(self.raw_file_names):
            fullpath = os.path.join(self.raw_dir, raw_path)
            with open(fullpath, 'r') as f:
                cfg = json.load(f)
            
            ## y (label)
            y = int(self.labels[raw_path.split('.')[0]]) - 1
     
            addr_to_id = dict() # {str: int}
            current_node_id = -1

            x = list() # node attributes
            seg = list() # node segment labels
            mask = list() # attention mask
            for addr, block in cfg.items(): # addr is 'str
                current_node_id += 1
                addr_to_id[addr] = current_node_id

                # get tokenized opcode sequence as node attributes
                tokenized_seq = []
                for insn in block['insn_list']:
                    opcode = insn['opcode']
                    operands = insn['operands']

                    opcode_operands = [opcode] + operands
                    normalized = self.normalize(opcode_operands)

                    tokenized = self.vocab.stoi.get(normalized, self.vocab.unk_index)
                    tokenized_seq.append(tokenized)

                # add [CLS] and [SEP]
                tokenized_seq = [self.vocab.sos_index] + tokenized_seq + [self.vocab.eos_index]
                
                # max seq len
                tokenized_seq = tokenized_seq[:self.seq_len]
                segment_label = [0 for _ in range(len(tokenized_seq))][:self.seq_len]
                attention_mask = [1 for _ in range(len(tokenized_seq))][:self.seq_len]

                # padding
                padding = [self.vocab.pad_index for _ in range(self.seq_len - len(tokenized_seq))]
                tokenized_seq.extend(padding); segment_label.extend(padding); attention_mask.extend(padding)
                
                x.append(tokenized_seq)
                seg.append(segment_label)
                mask.append(attention_mask)

            # construct dataset
            bbs_dataset = BBSDataset(x, seg, mask)
            bbs_dataloader = torch.utils.data.DataLoader(
                bbs_dataset, batch_size=100, shuffle=False, num_workers=5)
            
            bert_out = list()
            for bbs_data in bbs_dataloader:
                outputs = self.bert(
                    input_ids=bbs_data['inputs'].cuda(),
                    attention_mask=bbs_data['attention_mask'].cuda(),
                    token_type_ids=bbs_data['segment_labels'].cuda(),
                    return_dict=True     
                )

                bert_out.extend(outputs.pooler_output.tolist())
            
            # get sparse adjacent matrix
            edge_index = list()
            for addr, block in cfg.items(): # addr is `str`
                start_nid = addr_to_id[addr]
                for out_edge in block['out_edge_list']:
                    end_nid = addr_to_id[str(out_edge)]

                    ## edge_index
                    edge_index.append([start_nid, end_nid])
            
            # Data
            x = torch.tensor(bert_out)
            y = torch.tensor(y, dtype=torch.long).unsqueeze(0)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            data = Data(x=x, edge_index=edge_index, y=y)

            # save
            assert(self.raw_file_name_lookup(idx) == raw_path)
            save_path = 'data_{}_{}.pt'.format(idx, raw_path.split('.')[0])
            save_path = os.path.join(self.processed_dir, save_path)
            torch.save(data, save_path)

            idx += 1


class CFGDataset_MAGIC(CFGDataset):
    def __init__(self, root='/home/wubolun/data/malware/big2015/further'):

        # type of operators
        self.opcodeTypes = {'trans': 0, 'call': 1, 'math': 2, 'cmp': 3,
            'crypto': 4, 'mov': 5, 'term': 6, 'def': 7, 'other': 8}

        # type of operands
        self.operandTypes = {'num_const': len(self.opcodeTypes),
                     'str_const': len(self.opcodeTypes) + 1}

        # special characters
        self.specialChars = ['[', ']', '{', '}', '?', '@', '$']
        self.spChar2Idx = {val: idx for (idx, val) in enumerate(self.specialChars)}

        # instruction attributes length
        self.ins_dim = len(self.opcodeTypes) + len(self.operandTypes) + \
            + len(self.specialChars)

        # basic block structural feature
        self.vertexTypes = {'degree': self.ins_dim, 'num_inst': self.ins_dim + 1}

        # basic block attributes length
        self.block_dim = self.ins_dim + len(self.vertexTypes)

        super(CFGDataset_MAGIC, self).__init__(root)

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'cfg_magic')

    def process(self):
        ''' process raw JSON CFGs '''
        idx = 0
        for raw_path in tqdm.tqdm(self.raw_file_names):
            fullpath = os.path.join(self.raw_dir, raw_path)
            with open(fullpath, 'r') as f:
                cfg = json.load(f)

            # label
            y = int(self.labels[raw_path.split('.')[0]]) - 1

            addr_to_id = dict() # {str: int}
            current_node_id = -1

            x = list() # node attributes
            for addr, block in cfg.items():
                current_node_id += 1
                addr_to_id[addr] = current_node_id

                block_attr = self.get_block_attributes(block).tolist()
                x.append(block_attr)

            # get sparse adjacent matrix
            edge_index = list()
            for addr, block in cfg.items(): # addr is `str`
                start_nid = addr_to_id[addr]
                for out_edge in block['out_edge_list']:
                    end_nid = addr_to_id[str(out_edge)]

                    ## edge_index
                    edge_index.append([start_nid, end_nid])

            # Data
            x = torch.tensor(x)
            y = torch.tensor(y, dtype=torch.long).unsqueeze(0)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            data = Data(x=x, edge_index=edge_index, y=y)

            # save
            assert(self.raw_file_name_lookup(idx) == raw_path)
            save_path = 'data_{}_{}.pt'.format(idx, raw_path.split('.')[0])
            save_path = os.path.join(self.processed_dir, save_path)
            torch.save(data, save_path)

            idx += 1

    def get_block_attributes(self, block):
        ''' extract basic block attributes '''
        instAttr = np.zeros((1, self.ins_dim))
        for insn in block['insn_list']:
            attr = self.get_insn_attributes(insn)
            instAttr += np.array(attr)

        degree = len(block['in_edge_list']) + len(block['out_edge_list'])
        num_insn = len(block['insn_list'])
        return np.concatenate((instAttr, [degree, num_insn]), axis=None)

    def get_insn_attributes(self, insn):
        features_operand = [0] * len(self.operandTypes)
        for operand in insn['operands']:
            numeric_cnts, string_cnts = self._match_constant(operand)
            features_operand[0] += numeric_cnts
            features_operand[1] += string_cnts
        
        features_opcode = self._insn_opcode_features(insn)

        features_char = [0] * len(self.specialChars)
        insn_str = insn['opcode'] + ''.join(insn['operands'])
        for c in insn_str:
            if c in self.spChar2Idx:
                features_char[self.spChar2Idx[c]] += 1

        return features_operand + features_opcode + features_char
    
    def _insn_opcode_features(self, insn):

        features = [0] * len(self.opcodeTypes)
        opcode = insn['opcode']

        if opcode in CallingInstList:
            features[self.opcodeTypes['call']] += 1
        elif opcode in ConditionalJumpInstList:
            features[self.opcodeTypes['trans']] += 1
        elif opcode in UnconditionalJumpInstList:
            features[self.opcodeTypes['trans']] += 1
        elif opcode in EndHereInstList:
            features[self.opcodeTypes['term']] += 1
        elif opcode in RepeatInstList:
            features[self.opcodeTypes['trans']] += 1
            nested_features = self._insn_opcode_features({'opcode': insn['operands'][0]})
            features = [x + y for (x, y) in zip(features, nested_features)]
        elif opcode in MathInstList:
            features[self.opcodeTypes['math']] += 1
        elif opcode in CmpInstList:
            features[self.opcodeTypes['cmp']] += 1
        elif opcode in MovInstList:
            features[self.opcodeTypes['mov']] += 1
        elif opcode in DataInstList:
            features[self.opcodeTypes['def']] += 1
        else:
            features[self.opcodeTypes['other']] += 1
        
        return features

    def _match_constant(self, line):
        """Parse the numeric/string constants in an operand"""
        operand = line.strip('\n\r\t ')
        numericCnts = 0
        stringCnts = 0
        """
        Whole operand is a num OR leading num in expression.
        E.g. "0ABh", "589h", "0ABh" in "0ABh*589h"
        """
        wholeNum = r'^([1-9][0-9A-F]*|0[A-F][0-9A-F]*)h?.*'
        pattern = re.compile(wholeNum)
        if pattern.match(operand):
            numericCnts += 1
            # log.debug('[MatchConst] Match whole number in %s' % operand)
            # numerics.append('%s:WHOLE/LEAD' % operand)
        """Number inside expression, exclude the leading one."""
        numInExpr = r'([+*/:]|-)([1-9][0-9A-F]*|0[A-F][0-9A-F]*)h?'
        pattern = re.compile(numInExpr)
        match = pattern.findall(operand)
        if len(match) > 0:
            numericCnts += 1
            # log.debug('[MatchConst] Match in-expression number in %s' % operand)
            # numerics.append('%s:%d' % (operand, len(match)))
        """Const string inside double/single quote"""
        strRe = r'["\'][^"]+["\']'
        pattern = re.compile(strRe)
        match = pattern.findall(operand)
        if len(match) > 0:
            stringCnts += 1
            # log.debug('[MatchConst] Match str const in %s' % operand)
            # strings.append('%s:%d' % (operand, len(match)))

        return [numericCnts, stringCnts]


if __name__ == '__main__':

    # dataset = CFGDataset_Normalized_After_BERT(
    #     root='/home/wubolun/data/malware/big2015/further',
    #     vocab_path='/home/wubolun/data/malware/big2015/further/set_0.5_pair_30/normal.vocab',
    #     seq_len=64)
    dataset = CFGDataset_MAGIC(root='/home/wubolun/data/malware/big2015/further')
    print(len(dataset))

    for k in range(5):
        train_idx, val_idx = dataset.train_val_split(k)
        print('{}: train_len {}, val_len {}'.format(k, len(train_idx), len(val_idx)))

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for data in dataloader:
        print(data)
        print(data.x)
        print(data.x.shape)
        break
