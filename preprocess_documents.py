configuration = "fi-100_ft-15000000_nb-30_ht-100000_di-uni_mx-1500_lg-no"
dataset_path = "datasets"
model_path = "trained_models/pre_train_all_years_ded_coa_med_seq-128_old_" + configuration + "/pretrained_model"
year = 2017
suffix = "_pair_pre"

from data_loader import *

sentence_corpus, metadata_corpus = load_short_pairs(year, configuration, dataset_path)

from bert_sentence_pair_dataset import SentencePairDataset

sentence_data = SentencePairDataset(sentence_corpus)

from evaluate_bert import *
embeddings = get_embeddings(model_path, sentence_data, batch_size=512, resample=False, use_labels=False)

embedding_len = 512

host_embeddings = []
host_host_embeddings = []
hosts = []
host_hosts = []

host_host_embedding_idx = 0
for i, meta_doc in enumerate(metadata_corpus):
    host = meta_doc["ip_name"]
    hosts.append(host)
    all_embedding = np.zeros((embedding_len))
    num_doc_embeddings = 0
    for key, value in meta_doc["documents"].items():
        
        begin_idx = value["sentence_begin_idx"]
        end_idx = value["sentence_end_idx"]
        doc_embedding = np.zeros((embedding_len))
        for idx in range(begin_idx, end_idx):
            emb_idx = sentence_data.original_index[idx]
            doc_embedding += embeddings[emb_idx]
        value["embedding_idx"] = host_host_embedding_idx
        host_host_embedding_idx += 1
        host_hosts.append((host, key))
        host_host_embeddings.append(doc_embedding / (end_idx - begin_idx))
        all_embedding += doc_embedding
        num_doc_embeddings += 1
    if num_doc_embeddings == 0:
        print("No embeddings for host: " + host)
        continue
    all_embedding /= num_doc_embeddings
    host_embeddings.append(all_embedding)
    meta_doc["embedding_idx"] = i

host_embeddings = np.array(host_embeddings)
host_host_embeddings = np.array(host_host_embeddings)

import numpy as np
import json

save_path = "document_datasets/" + str(year) + suffix

import os
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.save(save_path + '/host_embeddings.npy', host_embeddings)
np.save(save_path + '/host_host_embeddings.npy', host_host_embeddings)
with open(save_path + '/metadata_corpus.json', 'w') as f:
    json.dump(metadata_corpus, f)
with open(save_path + '/hosts.txt', 'w') as f:
    for host in hosts:
        f.write(host + '\n')
with open(save_path + '/host_hosts.txt', 'w') as f:
    for host, doc in host_hosts:
        f.write(host + ',' + doc + '\n')