from torch.utils.data import Dataset, DataLoader
import random
import torch
from tqdm import tqdm

padding_token = 0
cls_token = 1
sep_token = 2
empty_token = 3
mask_token = 4

class BERTClassDataset(Dataset):
    def __init__(self, corpus, labels, seq_length, voc_min=None, voc_max=None, augment=False, shuffle=False, max_shift_amt=2):
        self.seq_len = seq_length
        self.conv_pairs = []
        self.conv_labels = []
    
        for document, label in tqdm(zip(corpus, labels)):
            
            # augment with probability 0.2 if label is 0
            prob = random.random()
            if label == 0 and augment and prob < 0.2:
                augmented_document = document.copy()
                shift_amt = random.randint(-max_shift_amt, max_shift_amt)
                for sentence in augmented_document:
                    for j in range(len(sentence)):
                        sentence[j] += shift_amt
                        sentence[j] = max(voc_min, min(voc_max, sentence[j]))
                for i in range(len(augmented_document)-1):
                    first = augmented_document[i][:(seq_length - 1) // 2]
                    second = augmented_document[i+1][:(seq_length - 1) // 2]
                    if not second:
                        second = [empty_token]
                    self.conv_pairs.append([first, second])
                    self.conv_labels.append(label)
                
            for i in range(len(document)-1):
                first = document[i][:(seq_length - 1) // 2]
                second = document[i+1][:(seq_length - 1) // 2]
                if not second:
                    second = [empty_token]
                self.conv_pairs.append([first, second])
                self.conv_labels.append(label)
        self.num_pairs = len(self.conv_pairs)
        
        if shuffle:
            import numpy as np
            indices = np.arange(self.num_pairs)
            np.random.shuffle(indices)
            self.conv_pairs = [self.conv_pairs[i] for i in indices]
            self.conv_labels = [self.conv_labels[i] for i in indices]
        

    def __len__(self):
        return self.num_pairs
    
    def __getitem__(self, idx):
        # Step 1: get the sentence pair
        s1, s2, label = self.get_pair(idx)

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
                  "token_type_ids": segment_ids, # "segment_label"
                  "label": label} # is_next
        return {key: torch.tensor(value) for key, value in output.items()}

    '''
    BERT Fine-Tuning is done on sentence pairs. This allows to capture slightly longer term features.
    '''
    def get_pair(self, index):
        s1, s2 = self.conv_pairs[index]
        label = self.conv_labels[index]
        return s1, s2, label