# created by Wu Bolun
# 2020.11.14
# bowenwu@sjtu.edu.cn

import torch
import torch.nn as nn

from transformers import BertConfig, BertModel, \
                         BertForPreTraining, BertForMaskedLM  


class PREBERT(nn.Module):
    ''' Pretraining BERT Model 
    Ref: https://huggingface.co/transformers/model_doc/bert.html#transformers.BertForPreTraining
    '''
    def __init__(self, config:BertConfig):
        super().__init__()
        self.config = config
        self.bert = BertForPreTraining(self.config)

    def from_pretrained(self, path):
        self.bert.load_state_dict(torch.load(path))

    def forward(self, x, attention_mask, token_type_ids, 
                mlm_labels=None, nsp_label=None):
        ''' 
        loss is calculated by CrossEntropyLoss
        which is a combination of LogSoftmax and NLLLoss
        So there is no need to add LogSoftmax at the end of the network
        '''
        out = self.bert(input_ids=x,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=mlm_labels,
                        next_sentence_label=nsp_label,
                        return_dict=True)

        return out.loss, out.seq_relationship_logits, out.prediction_logits, 


class MLMBERT(nn.Module):
    ''' BERT for Masked LM task
    Ref: https://huggingface.co/transformers/model_doc/bert.html#transformers.BertForMaskedLM
    '''
    def __init__(self, config:BertConfig):
        super().__init__()
        self.config = config
        self.bert = BertForMaskedLM(self.config)
    
    def from_pretrained(self, path):
        self.bert.load_state_dict(torch.load(path))
    
    def forward(self, x, attention_mask, token_type_ids,
                mlm_labels):
        out = self.bert(input_ids=x,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=mlm_labels,
                        return_dict=True)
        return out.loss, out.logits
