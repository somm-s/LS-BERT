from torch.utils.data import Dataset, DataLoader
import random
import torch
import numpy as np
from tqdm import tqdm
import re
from pre_train_bert import get_buckets, get_uni_directional, get_voc_size
import numpy as np

padding_token = 0
cls_token = 1
sep_token = 2
empty_token = 3
mask_token = 4
time_token = 5
time_repetition_token = 8
token_repetition_token = 9
coalesce_min_token = 8
normal_min_token = 10

def tokenize_sentence_pair(sentence_pair):
    s1, s2, ip1, ip2, start_time, end_time = sentence_pair
    tokens1 = [int(token) for token in s1.split()]
    tokens2 = [int(token) for token in s2.split()]
    tokens1 = np.array(tokens1, dtype=np.uint8)
    tokens2 = np.array(tokens2, dtype=np.uint8)
    return tokens1, tokens2, ip1, ip2, start_time, end_time

class BERTInferenceDataset(Dataset):
    def __init__(self, sentence_pairs, seq_length=128):
        print("Creating BERT Inference Dataset")
        self.seq_len = seq_length
        self.conv_pairs = []
        self.metadata = []
        for sentence_pair in tqdm(sentence_pairs):
            s1, s2, ip1, ip2, start_time, end_time = sentence_pair
            sentence_pair = s1[:seq_length * 3], s2[:seq_length * 3], ip1, ip2, start_time, end_time
            s1, s2, ip1, ip2, start_time, end_time = tokenize_sentence_pair(sentence_pair)
            self.conv_pairs.append((s1, s2))
            self.metadata.append((ip1, ip2, start_time, end_time))
        self.num_pairs = len(self.conv_pairs)

    def __len__(self):
        return self.num_pairs
    
    def __getitem__(self, idx):
        return self.get_pair(idx)
                
    def get_pair(self, idx):
        s1, s2 = self.conv_pairs[idx]
        
        # Step 2: truncate the sentences to the maximum length / 2 - 2
        s1 = list(s1[:self.seq_len // 2 - 2])
        s2 = list(s2[:self.seq_len // 2 - 2])

        # Step 3: Adding CLS and SEP tokens to the start and end of sentences
        # Adding PAD token for labels
        t1 = [cls_token] + s1 + [sep_token]
        t2 = s2 + [sep_token]
        
        # Step 4: combine sentence 1 and 2 as one input
        # adding PAD tokens to make the sentence same length as seq_len
        segment_ids = ([0 for _ in range(len(t1))] + [1 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        attention_mask = [1 for _ in range(len(bert_input))]
        padding = [padding_token for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), segment_ids.extend(padding), attention_mask.extend(padding)

        output = {"input_ids": bert_input, # "bert_input"
                  "attention_mask": attention_mask, # "attention_mask"
                  "token_type_ids": segment_ids,} # "segment_label"
                #   "labels": meta_dict} # is_next
        return {key: torch.tensor(value) for key, value in output.items()}