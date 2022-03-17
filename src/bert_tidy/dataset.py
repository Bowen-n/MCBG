from torch.utils.data import Dataset
import tqdm
import torch
import random
import argparse


from vocab import WordVocab


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, ignored_label=0,
                encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.ignored_label = ignored_label
        self.encoding = encoding

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item) # 50% continuous sentences
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.ignored_label] + t1_label + [self.ignored_label]
        t2_label = t2_label + [self.ignored_label]

        segment_label = ([0 for _ in range(len(t1))] + [1 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]
        attention_mask = ([1 for _ in range(len(t1) + len(t2))])[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), segment_label.extend(padding), attention_mask.extend(padding)

        padding_label = [self.ignored_label for _ in range(self.seq_len - len(bert_label))]
        bert_label.extend(padding_label)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "attention_mask": attention_mask,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index) # if not exist, return UNK TOKEN

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(self.ignored_label)

        return tokens, output_label

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(next:0, not_next:1)
        # check https://huggingface.co/transformers/model_doc/bert.html
        if random.random() > 0.5:
            return t1, t2, 0
        else:
            return t1, self.get_random_line(), 1

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_path', type=str)
    parser.add_argument('-v', '--val_path', type=str)
    parser.add_argument('-c', '--vocab_path', type=str)

    args = parser.parse_args()
    train_path = args.train_path
    val_path = args.val_path
    vocab_path = args.vocab_path

    # train_path = '/home/wubolun/data/malware/big2015/bbs_corpus_normal.small'
    # val_path = '/home/wubolun/data/malware/big2015/bbs_corpus_normal_val.small'
    # vocab_path = '/home/wubolun/data/malware/big2015/bbs_vocab_normal.small'
    BATCH_SIZE = 256

    # dataset
    vocab = WordVocab.load_vocab(vocab_path)
    train_dataset = BERTDataset(train_path, vocab, seq_len=64, ignored_label=-100)
    val_dataset = BERTDataset(val_path, vocab, seq_len=64, ignored_label=-100)
    print('corpus size: train {}, val {}.'.format(len(train_dataset), len(val_dataset)))
    print('vocab: {}'.format(len(vocab)))
    dataloader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE)
    for data in dataloader:
        print(data['bert_input'].shape)
        break

    