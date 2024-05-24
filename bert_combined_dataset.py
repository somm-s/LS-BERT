from torch.utils.data import Dataset, DataLoader
import random
import torch
import numpy as np
from tqdm import tqdm
import re
from pre_train_bert import get_buckets, get_uni_directional, get_voc_size

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


# coalescing repetitive tokens
token_pattern = r'\b(\d\d)(?:\s+\1){4,}\b'
token_replacement = r'\1 9'

# coalescing repetitive time tokens
time_pattern = r'\b(\d)(?:\s+\1){4,}\b'
time_replacement = r'\1 8'



def coalesce_sentences(corpus):
    coalesced_corpus = []
    for host_host_doc in tqdm(corpus):
        coalesced_doc = []
        for i in range(len(host_host_doc)):
            token_replaced = re.sub(token_pattern, token_replacement, host_host_doc[i])
            time_replaced = re.sub(time_pattern, time_replacement, token_replaced)
            coalesced_doc.append(time_replaced)
        coalesced_corpus.append(coalesced_doc)
    return coalesced_corpus

def create_sentence_pairs(corpus, outlist, seq_length=None):
    if seq_length:
        seq_length = (3 * seq_length) // 2
    else:
        seq_length = 5000 # sentences longer are truncated
    
    for host_host_doc in tqdm(corpus):
        for i in range(len(host_host_doc) - 1):
            first = host_host_doc[i][:seq_length]
            second = host_host_doc[i + 1][:seq_length]
            outlist.append((first, second))
        outlist.append((host_host_doc[-1], "3"))
        
def token_from_sentence_pairs(sentence_pairs):
    sentence_pairs_np = []
    for i in tqdm(range(len(sentence_pairs))):
        s1, s2 = sentence_pairs[i]
        tokens1 = [int(token) for token in s1.split()]
        tokens2 = [int(token) for token in s2.split()]
        tokens1 = np.array(tokens1, dtype=np.uint8)
        tokens2 = np.array(tokens2, dtype=np.uint8)
        sentence_pairs_np.append((tokens1, tokens2))
    return sentence_pairs_np


class FlowPairDataset(Dataset):
    def __init__(self, config, corpus_0, corpus_1=None, seq_length=512, deduplicate=False, coalesce=False, shuffle=True, balanced=False):
        print("Creating dataset for ", config)
        if coalesce:
            self.min_token = coalesce_min_token
        else:
            self.min_token = normal_min_token
        self.max_token = get_voc_size(config) - 1
        self.seq_len = seq_length
        self.conv_pairs_pre = []
        self.conv_pairs_0 = []
        self.conv_pairs_1 = []
        self.is_finetune = corpus_1 is not None
        
        # Step 0: convert to same format
        # Step 1: create sentence pairs (using concatenation?)
        # Step 2: deduplicate pairs --> hash and skip if same hash
        # Step 3: process to real tokens
        # Step 4: coalesce
        # Step 5: deduplicate again (if deduplicate)
        
        # Step 0: convert to same format
        # Structure if corpus_1 given (dataset for finetuning): corpus [ ext_ip_doc [ int_ip_doc [ s1, s2, ... ], ... ], ... ]
        # Structure if corpus_1 not given (dataset for pretraining): corpus [ host_host_doc [ s1, s2, ... ], ... ]
        if self.is_finetune:
            
            # check whether normalization is necessary (if corpus_0 consists of lists of strings)
            # check whether normalization is necessary (if corpus_0 consists of lists of strings)
            if isinstance(corpus_0[0][0], list):
                print("Normalizing corpus 0")
                corpus_0_conv = []
                for ext_ip_doc in corpus_0:
                    for int_ip_doc in ext_ip_doc:
                        corpus_0_conv.append(int_ip_doc)
            else:
                corpus_0_conv = corpus_0

            if corpus_1 and isinstance(corpus_1[0][0], list):
                print("Normalizing corpus 1")
                corpus_1_conv = []
                for ext_ip_doc in corpus_1:
                    for int_ip_doc in ext_ip_doc:
                        corpus_1_conv.append(int_ip_doc)
            elif corpus_1:
                corpus_1_conv = corpus_1
        else:
            corpus_0_conv = corpus_0
            
        # Step 1: Coalesce using regex
        if coalesce:
            print("Coalescing")
            corpus_0_conv = coalesce_sentences(corpus_0_conv)
            if self.is_finetune:
                corpus_1_conv = coalesce_sentences(corpus_1_conv)
            
        # Step 2: create sentence pairs (using concatenation?)
        if self.is_finetune:
            print("Creating sentence pairs")
            create_sentence_pairs(corpus_0_conv, self.conv_pairs_0, seq_length=seq_length)
            create_sentence_pairs(corpus_1_conv, self.conv_pairs_1, seq_length=seq_length)
            print("Training pairs 0:", len(self.conv_pairs_0))
            print("Training pairs 1:", len(self.conv_pairs_1))
        else:
            create_sentence_pairs(corpus_0_conv, self.conv_pairs_pre, seq_length=seq_length)
            print("Pretraining pairs:", len(self.conv_pairs_pre))
            
        
            
        # Step 3: deduplicate pairs --> hash and skip if same hash
        if deduplicate:
            print("Deduplicating")
            if self.is_finetune:
                self.conv_pairs_0 = list(set(self.conv_pairs_0))
                self.conv_pairs_1 = list(set(self.conv_pairs_1))
                print("Training pairs after deduplication (0):", len(self.conv_pairs_0))
                print("Training pairs after deduplication (1):", len(self.conv_pairs_1))
            else:
                self.conv_pairs_pre = list(set(self.conv_pairs_pre))
                print("Pretraining pairs after deduplication:", len(self.conv_pairs_pre))
            
        
        # Step 4: process to real tokens
        print("Extracting tokens")
        self.conv_pairs_pre = token_from_sentence_pairs(self.conv_pairs_pre)
        self.conv_pairs_0 = token_from_sentence_pairs(self.conv_pairs_0)
        self.conv_pairs_1 = token_from_sentence_pairs(self.conv_pairs_1)
        
        if self.is_finetune:
            if balanced:
                print("Balancing")
                len_0 = len(self.conv_pairs_0)
                len_1 = len(self.conv_pairs_1)
                min_len = min(len_0, len_1)
                # shuffle and truncate
                indices_0 = np.arange(len_0)
                np.random.shuffle(indices_0)
                indices_1 = np.arange(len_1)
                np.random.shuffle(indices_1)
                self.conv_pairs_0 = [self.conv_pairs_0[i] for i in indices_0[:min_len]]
                self.conv_pairs_1 = [self.conv_pairs_1[i] for i in indices_1[:min_len]]
            
            self.conv_pairs = self.conv_pairs_0 + self.conv_pairs_1
            self.conv_labels = [0 for _ in range(len(self.conv_pairs_0))] + [1 for _ in range(len(self.conv_pairs_1))]        
        else:
            self.conv_pairs = self.conv_pairs_pre
            self.conv_labels = [0 for _ in range(len(self.conv_pairs))] # dummy labels
        
        self.num_pairs = len(self.conv_pairs)
        if shuffle:
            indices = np.arange(self.num_pairs)
            np.random.shuffle(indices)
            self.conv_pairs = [self.conv_pairs[i] for i in indices]
            self.conv_labels = [self.conv_labels[i] for i in indices]

    def __len__(self):
        return self.num_pairs
    
    def __getitem__(self, idx):
        if self.is_finetune:
            return self.get_finetune_pair(idx)
        else:
            return self.get_masked_pair(idx)
        
    def get_finetune_pair(self, idx):
        # Step 1: get the sentence pair
        s1, s2, label = self.get_f_pair(idx)
        
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
                  "token_type_ids": segment_ids, # "segment_label"
                  "label": label} # is_next
        return {key: torch.tensor(value) for key, value in output.items()}

    
    def get_masked_pair(self, idx):
        # Step 1: get a random sentence pair, either negative or positive (saved as is_next_label)
        #         is_next=1 means the second sentence comes after the first one in the conversation.
        s1, s2, is_random = self.get_m_pair(idx)
        
        # Truncate the sentences
        s1 = s1[:self.seq_len // 2 - 2]
        s2 = s2[:self.seq_len // 2 - 2]

        # Step 2: replace random words in sentence with mask / random words
        masked_numericalized_s1, s1_mask = self.mask_sentence(s1)
        masked_numericalized_s2, s2_mask = self.mask_sentence(s2)

        # Step 3: Adding CLS and SEP tokens to the start and end of sentences
        # Adding PAD token for labels
        t1 = [cls_token] + masked_numericalized_s1 + [sep_token]
        t2 = masked_numericalized_s2 + [sep_token]
        t1_mask = [padding_token] + s1_mask + [padding_token]
        t2_mask = s2_mask + [padding_token]

        # Step 4: combine sentence 1 and 2 as one input
        # adding PAD tokens to make the sentence same length as seq_len
        segment_ids = ([0 for _ in range(len(t1))] + [1 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_mask + t2_mask)[:self.seq_len]
        
        # labels is equal to bert_input except for where bert_label has a non-zero value
        # in which case it is equal to bert_label
        labels = [bert_label[i] if bert_label[i] != 0 else bert_input[i] for i in range(len(bert_input))] # TODO: -100 values
        attention_mask = [1 for _ in range(len(bert_input))]
        
        padding = [padding_token for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_ids.extend(padding), labels.extend(padding), attention_mask.extend(padding)

        output = {"input_ids": bert_input, # "bert_input"
                  "attention_mask": attention_mask, # "attention_mask"
                  "labels": labels, # "bert_label"
                  "token_type_ids": segment_ids, # "segment_label"
                  "next_sentence_label": is_random} # is_next
        return {key: torch.tensor(value) for key, value in output.items()}

    '''
    BERT Fine-Tuning is done on sentence pairs. This allows to capture slightly longer term features.
    '''
    def get_f_pair(self, index):
        s1, s2 = self.conv_pairs[index]
        label = self.conv_labels[index]
        return s1, s2, label

    '''
    BERT Training makes use of the following two strategies:
    1. Next Sentence Prediction (NSP)
    During training the model gets as input pairs of sentences and it learns to predict if the second sentence is the next sentence in the original text as well.
    During training the model is fed with two input sentences at a time such that:
        - 50% of the time the second sentence comes after the first one.
        - 50% of the time it is a a random sentence from the full corpus.
    BERT is then required to predict whether the second sentence is random or not, with the assumption that the random sentence will be disconnected from the first sentence:
    '''
    def get_m_pair(self, index):
        s1, s2 = self.conv_pairs[index]
        is_random = 0
        if random.random() > 0.5:
            random_index = random.randrange(len(self.conv_pairs))
            s2 = self.conv_pairs[random_index][1]
            is_random = 1
        return s1, s2, is_random

    '''
    2. Masked LM (MLM): 
    Randomly mask out 15% of the words in the input — replacing them with a [MASK] token.
    Run the entire sequence through the BERT attention based encoder and then predict only the masked words, based on the context provided by the other non-masked words in the sequence. However, there is a problem with this naive masking approach — the model only tries to predict when the [MASK] token is present in the input, while we want the model to try to predict the correct tokens regardless of what token is present in the input. To deal with this issue, out of the 15% of the tokens selected for masking:
        80% of the tokens are actually replaced with the token [MASK].
        10% of the time tokens are replaced with a random token.
        10% of the time tokens are left unchanged.
    '''
    def mask_sentence(self, s):
        adapted_s = []
        mask = []
        for token in s:
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    adapted_s.append(mask_token)
                elif prob < 0.9:
                    random_token = random.randint(self.min_token, self.max_token)
                    while random_token == token:
                        random_token = random.randint(self.min_token, self.max_token)
                    adapted_s.append(random_token)
                else:
                    adapted_s.append(token)
                mask.append(token)
            else:
                adapted_s.append(token)
                mask.append(0)
        return adapted_s, mask