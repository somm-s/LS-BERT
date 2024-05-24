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
        s = corpus[i][:10000]
        token_replaced = re.sub(token_pattern, token_replacement, s)
        time_replaced = re.sub(time_pattern, time_replacement, token_replaced)
        coalesced_corpus.append(time_replaced[:seq_length * 3])
    return coalesced_corpus
        
def token_from_sentence(sentences, seq_length=128):
    sentences_np = []
    for i in tqdm(range(len(sentences))):
        s = sentences[i]
        tokens = [int(token) for token in s.split()]
        tokens = tokens[:seq_length - 2]
        tokens = np.array(tokens, dtype=np.uint8)
        sentences_np.append(tokens)
    return sentences_np


class SingleSentenceDataset(Dataset):
    def __init__(self, sentence_corpus, seq_length=128, coalesce=True):
        self.seq_len = seq_length
        
        self.original_index = []         
        self.original_sentences = {}
        self.sentences = []
        sentence_index = 0
        for i in tqdm(range(len(sentence_corpus))):
            if sentence_corpus[i] not in self.original_sentences:
                self.original_sentences[sentence_corpus[i]] = sentence_index
                self.original_index.append(sentence_index)
                self.sentences.append(sentence_corpus[i])
                sentence_index += 1
            else:
                self.original_index.append(self.original_sentences[sentence_corpus[i]])
                                
        if coalesce:
            print("Coalescing")
            self.sentences = coalesce_sentences(self.sentences)
        
        print("Extracting tokens")
        self.sentences = token_from_sentence(self.sentences)
        

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return self.get_sentence(idx)
        
    def get_sentence(self, idx):
        s = self.sentences[idx]
        s = list(s[:self.seq_len - 2])
        t = [cls_token] + s + [sep_token]
        segment_ids = ([0 for _ in range(len(t))])
        attention_mask = [1 for _ in range(len(t))]
        padding = [padding_token for _ in range(self.seq_len - len(t))]
        t.extend(padding), segment_ids.extend(padding), attention_mask.extend(padding)
        output =  {"input_ids": t, "attention_mask": attention_mask, "token_type_ids": segment_ids}
        return {k: torch.tensor(v) for k, v in output.items()}
