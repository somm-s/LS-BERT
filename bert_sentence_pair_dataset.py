from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from tqdm import tqdm
import re

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

def coalesce_sentences(corpus, seq_length=128):
    coalesced_corpus = []
    for i in tqdm(range(len(corpus))):
        s1, s2 = corpus[i]
        s1, s2 = s1[:5000], s2[:5000]
        token_replaced_1 = re.sub(token_pattern, token_replacement, s1)
        token_replaced_2 = re.sub(token_pattern, token_replacement, s2)
        time_replaced_1 = re.sub(time_pattern, time_replacement, token_replaced_1)
        time_replaced_2 = re.sub(time_pattern, time_replacement, token_replaced_2)
        coalesced_corpus.append((time_replaced_1[:seq_length * 3], time_replaced_2[:seq_length * 3]))
    return coalesced_corpus
        
def token_from_sentence(sentences, seq_length=128):
    sentences_np = []
    for i in tqdm(range(len(sentences))):
        s1, s2 = sentences[i]
        tokens_1 = [int(token) for token in s1.split()][:seq_length - 2]
        tokens_2 = [int(token) for token in s2.split()][:seq_length - 2]
        tokens_1 = np.array(tokens_1, dtype=np.uint8)
        tokens_2 = np.array(tokens_2, dtype=np.uint8)
        tokens = (tokens_1, tokens_2)
        sentences_np.append(tokens)
    return sentences_np


class SentencePairDataset(Dataset):
    def __init__(self, sentence_pair_corpus, seq_length=128, coalesce=True):
        
        # structure of sentence_pair_corpus: [(sentence1, sentence2), ...] where all sentence pairs are contiguous
        
        self.seq_len = seq_length
        
        self.original_index = []         
        self.original_sentences = {}
        self.sentences = []
        sentence_index = 0
        for i in tqdm(range(len(sentence_pair_corpus))):
            if sentence_pair_corpus[i] not in self.original_sentences:
                self.original_sentences[sentence_pair_corpus[i]] = sentence_index
                self.original_index.append(sentence_index)
                self.sentences.append(sentence_pair_corpus[i])
                sentence_index += 1
            else:
                self.original_index.append(self.original_sentences[sentence_pair_corpus[i]])
                                
        if coalesce:
            print("Coalescing")
            self.sentences = coalesce_sentences(self.sentences)
        
        print("Extracting tokens")
        self.sentences = token_from_sentence(self.sentences)
        

    def __len__(self):
        return len(self.original_index)
    
    def __getitem__(self, idx):
        return self.get_sentence(self.original_index[idx])
        
    def get_sentence(self, idx):
        s1, s2 = self.sentences[idx]
        s1, s2 = list(s1[:self.seq_len // 2 - 2]), list(s2[:self.seq_len // 2 - 2])
        t = [cls_token] + s1 + [sep_token] + s2 + [sep_token]
        segment_ids = ([0 for _ in range(len(s1) + 2)] + [1 for _ in range(len(s2) + 1)])
        attention_mask = [1 for _ in range(len(t))]
        padding = [padding_token for _ in range(self.seq_len - len(t))]
        t.extend(padding), segment_ids.extend(padding), attention_mask.extend(padding)
        output =  {"input_ids": t, "attention_mask": attention_mask, "token_type_ids": segment_ids}
        return {k: torch.tensor(v) for k, v in output.items()}
