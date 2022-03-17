# created by Wu Bolun
# 2020.11.11
# bowenwu@sjtu.edu.cn

import os
import re
import json
import random
import tqdm
import csv
import argparse


# ref :https://blog.csdn.net/vhelsing/article/details/4283491
REGS = [
    'eax', 'ebx', 'ecx', 'edx',
    'ebp', 'esp', 'bp', 'sp',
    'edi', 'esi', 'si', 'di',
    'es', 'cs', 'ss', 'ds', 'fs', 'gs',
    'eip', 'ip',
    'cf', 'pf', 'af', 'zf', 'sf', 'of', 
    'tf', 'if', 'df',
    'ax', 'bx', 'cx', 'dx',
    'ah', 'al', 'bh', 'bl', 'ch', 'cl', 'dh', 'dl',
]

TYPES = [
    'dword', 'short', 'byte', 'offset', 'ptr'
]


class BertDatasetCreator(object):
    ''' Create dataset for BERT training '''
    def __init__(self, data_dir, storepath, corpus_files_idx):
        self.data_dir = data_dir
        self.file_list = []
        for i, filename in enumerate(os.listdir(self.data_dir)):
            if i in corpus_files_idx:
                self.file_list.append(filename)
        self.file_iter = tqdm.tqdm(self.file_list)
        self.storepath = storepath
    

    def opcode_format(self):
        ''' use opcode to represent an instruction '''
        store_f = open(self.storepath, 'a+')

        for path in self.file_iter:
            # print('File: {}'.format(path))
            fullpath = os.path.join(self.data_dir, path)
            with open(fullpath, 'r') as f:
                cfg = json.load(f)

            addr_used = []

            for addr, block in cfg.items():
                if addr in addr_used or \
                len(block['out_edge_list']) == 0:
                    continue

                first_seq = []
                for insn in block['insn_list']:
                    first_seq.append(insn['opcode'])
                
                # check if second sequence exists
                sec_addr = None
                for out_edge in block['out_edge_list']:
                    if str(out_edge) not in addr_used and str(out_edge) != addr:
                        sec_addr = str(out_edge)
                if sec_addr is None:
                    continue

                second_seq = []
                for insn in cfg[sec_addr]['insn_list']:
                    second_seq.append(insn['opcode'])

                if len(first_seq) + len(second_seq) > 20:
                    continue
                if len(first_seq) == 0 or len(second_seq) == 0:
                    continue
                addr_used.append(addr)
                addr_used.append(sec_addr)
                
                for opcode in first_seq:
                    store_f.write(opcode+' ')
                store_f.write('\t')
                for opcode in second_seq:
                    store_f.write(opcode+' ')
                store_f.write('\n')

        store_f.close()
    

    def normalized_format(self, pairs_count=30):
        ''' normalize instruction '''
        store_f = open(self.storepath, 'a+')

        count_64 = 0
        count_total = 0

        for path in self.file_iter:
            fullpath = os.path.join(self.data_dir, path)
            with open(fullpath, 'r') as f:
                cfg = json.load(f)
            

            all_insns = []
            all_insns_normal = []
            count = 0

            key_list = list(cfg.keys())
            random.shuffle(key_list)
            # for addr, block in cfg.items():
            for addr in key_list:
                block = cfg[addr]
                # addr is str
                if len(block['out_edge_list']) == 0:
                    continue
                
                # first block
                first_seq = [] # str list
                first_seq_norm = []
                for insn in block['insn_list']:
                    opcode = insn['opcode']
                    operands = insn['operands']

                    # seq_str
                    operands_str = ', '.join(operands)
                    first_seq.append(opcode + ' ' + operands_str)

                    # seq_norm_str
                    opcode_operands = [opcode] + operands
                    first_seq_norm.append(self.normalize(opcode_operands))

                sec_addr = None
                for out_edge in block['out_edge_list']:
                    if str(out_edge) != addr:
                        sec_addr = str(out_edge)
                        break
                if sec_addr is None:
                    continue
                
                second_seq = []
                second_seq_norm = []

                for insn in cfg[sec_addr]['insn_list']:
                    opcode = insn['opcode']
                    operands = insn['operands']

                    operands_str = ', '.join(operands)
                    second_seq.append(opcode + ' ' + operands_str)

                    opcode_operands = [opcode] + operands
                    second_seq_norm.append(self.normalize(opcode_operands))

                if len(first_seq_norm) + len(second_seq_norm) > 64:
                    count_64 += 1
                    continue
                else:
                    count_total += 1

                if len(first_seq_norm) == 0 or len(second_seq_norm) == 0:
                    continue

                # print('{}\t{}'.format(first_seq, first_seq_norm))

                for ins in first_seq_norm:
                    store_f.write(ins+' ')
                store_f.write('\t')
                for ins in second_seq_norm:
                    store_f.write(ins+' ')
                store_f.write('\n')

                count += 1
                if count == pairs_count:
                    break

            # end one file
        # end all files
        store_f.close()
        # print(count_total)
        # print(count_64)


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
        [ebp-1Ch] [ebp+8+4] [esp+18h+-4] [esp+40h+-18h]
        [ebp+esp*4]
        [ebp + 8]
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


def corpus_file_select(data_dir, labels, mode):
    label_list = list()
    file_list = os.listdir(data_dir)
    random.shuffle(file_list)
    for filename in os.listdir(data_dir):
        label = labels[filename.split('.')[0]]
        label_list.append(int(label))

    groups = []
    for _ in range(9):
        groups.append([])
    for i, label in enumerate(label_list):
        groups[label-1].append(i)
    

    corpus_files = []
    for group in groups:
        if mode == 'train':
            corpus_files.extend(group[:int(len(group)*0.5)])
        elif mode =='val':
            corpus_files.extend(group[int(len(group)*0.95):])
    return corpus_files


if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', type=str)
    parser.add_argument('-l', '--label_path', type=str)
    parser.add_argument('-s', '--store_path', type=str)
    parser.add_argument('-m', '--mode', type=str, default='train')

    args = parser.parse_args()
    data_dir = args.data_dir
    label_path = args.label_path
    store_path = args.store_path
    mode = args.mode

    # data_dir = '/home/wubolun/data/malware/big2015/cfg'
    # label_path = '/home/wubolun/data/malware/big2015/trainLabels.csv'
    # store_path = '/home/wubolun/data/malware/big2015/bbs_corpus_normal_val.small'

    labels = {}
    with open(label_path, 'r') as f:
        data = csv.reader(f)
        for row in data:
            if row[0] == 'Id':
                continue
            labels[row[0]] = int(row[1])
    
    corpus_files_idx = corpus_file_select(data_dir, labels, mode)
    print(len(corpus_files_idx))

    bert_dataset_creator = BertDatasetCreator(data_dir, store_path, corpus_files_idx)
    bert_dataset_creator.normalized_format()

