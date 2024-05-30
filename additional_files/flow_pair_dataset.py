from torch.utils.data import Dataset, DataLoader
import random
import torch
from tqdm import tqdm

padding_token = 0
cls_token = 1
sep_token = 2
empty_token = 3
mask_token = 4
time_token = 5

def sentence_to_token_array(sentence, seq_length):
    half_seq_len = (seq_length - 1) // 2
    tokens = sentence.split(" ")
    tokens = tokens[:half_seq_len]
    return tokens
    

class FlowPairDataset(Dataset):
    def __init__(self, corpus_0, corpus_1=None, seq_length=512, deduplicate=False, coalesce=False, shuffle=True):
        self.seq_len = seq_length
        self.conv_pairs_pre = []
        self.conv_pairs_0 = []
        self.conv_pairs_1 = []
        
        if corpus_1:
            corpus_0 = [[sentence_to_token_array(sentence) for sentence in document] for document in corpus_0]
            corpus_1 = [[sentence_to_token_array(sentence) for sentence in document] for document in corpus_1]
            
            
            

        self.num_pairs = len(self.conv_pairs)
        if shuffle:
            random.shuffle(self.conv_pairs)

    def __len__(self):
        return self.num_pairs
    
    def __getitem__(self, idx):
        # Step 1: get a random sentence pair, either negative or positive (saved as is_next_label)
        #         is_next=1 means the second sentence comes after the first one in the conversation.
        s1, s2, is_random = self.get_pair(idx)

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
    BERT Training makes use of the following two strategies:
    1. Next Sentence Prediction (NSP)
    During training the model gets as input pairs of sentences and it learns to predict if the second sentence is the next sentence in the original text as well.
    During training the model is fed with two input sentences at a time such that:
        - 50% of the time the second sentence comes after the first one.
        - 50% of the time it is a a random sentence from the full corpus.
    BERT is then required to predict whether the second sentence is random or not, with the assumption that the random sentence will be disconnected from the first sentence:
    '''
    def get_pair(self, index):
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
                    random_token = random.randint(self.min_bucket, self.max_bucket)
                    while random_token == token:
                        random_token = random.randint(self.min_bucket, self.max_bucket)
                    adapted_s.append(random_token)
                else:
                    adapted_s.append(token)
                mask.append(token)
            else:
                adapted_s.append(token)
                mask.append(0)
        return adapted_s, mask