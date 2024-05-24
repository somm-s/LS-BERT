

import joblib
import numpy as np
from similarity_dataset import *
import os
import time
from bert_sentence_pair_dataset import *
from evaluate_bert import *
import pandas as pd

def get_next_sentences(document_path, ip_documents):    
    file_list = os.listdir(document_path)
    ip1_list = []
    ip2_list = []
    sentence_list = []
    for file in file_list:
        with open(document_path + "/" + file, 'r') as f:
            ip1 = file.split(".txt")[0]
            if ip1 in ip_documents:
                position = ip_documents[ip1]["position"]
                f.seek(position)
                lines = f.readlines()
                ip_documents[ip1]["position"] = f.tell()
            else:
                lines = f.readlines()
                ip_documents[ip1] = {"position": f.tell()}
                        
            for line in lines:
                line.strip()
                ip2, start_time, end_time, sentence = line.split(",")
                ip1_list.append(ip1)
                ip2_list.append(ip2)
                sentence_list.append(sentence)
    return ip1_list, ip2_list, sentence_list


def update_embeddings(sentence_dataset, metadata_corpus, ip1_list, ip2_list, embeddings):
    changed_ip1_hosts = set()
    changed_ip2_hosts = set()
    for i, (ip1, ip2) in enumerate(zip(ip1_list, ip2_list)):
        emb_idx = sentence_dataset.original_index[i]
        embedding = embeddings[emb_idx]
        
        if ip1 not in metadata_corpus:
            metadata_corpus[ip1] = {
                "embedding" : embedding,
                "documents" : {
                    ip2 : embedding,
                },
                "documents_meta": {
                    ip2 : {}
                },
                "counts" : {
                    ip2 : 1
                }
            }
            changed_ip1_hosts.add(ip1)
            changed_ip2_hosts.add(ip1 + "-" + ip2)
        else:
            if ip2 not in metadata_corpus[ip1]["documents"]:
                metadata_corpus[ip1]["documents_meta"][ip2] = {}
                metadata_corpus[ip1]["documents"][ip2] = embedding
                metadata_corpus[ip1]["counts"][ip2] = 1
                
            else:
                n = metadata_corpus[ip1]["counts"][ip2]
                old_emb = metadata_corpus[ip1]["documents"][ip2]
                new_emb = (old_emb * n + embedding)
                metadata_corpus[ip1]["documents"][ip2] = new_emb / (n + 1)
                metadata_corpus[ip1]["counts"][ip2] = n + 1
                
            changed_ip2_hosts.add(ip1 + "-" + ip2)  
            changed_ip1_hosts.add(ip1)
                
            doc_emb = np.zeros(embedding.shape)
            for ip2 in metadata_corpus[ip1]["documents"]:
                doc_emb += metadata_corpus[ip1]["documents"][ip2]
            doc_emb /= len(metadata_corpus[ip1]["documents"])
            metadata_corpus[ip1]["embedding"] = doc_emb
    class_host_embeddings = []
    class_host_host_embeddings = []
    class_ip1 = []
    class_ip2 = [] 
    for ip1 in changed_ip1_hosts:
        class_ip1.append(ip1)
        class_host_embeddings.append(metadata_corpus[ip1]["embedding"])
    for ip1_ip2 in changed_ip2_hosts:
        ip1, ip2 = ip1_ip2.split("-")
        class_ip2.append((ip1, ip2))
        class_host_host_embeddings.append(metadata_corpus[ip1]["documents"][ip2])
    return class_host_embeddings, class_host_host_embeddings, class_ip1, class_ip2


def get_similarity_batch(embeddings, similarity_dataset, is_host_host=False, batch_size=512, top_k=5):

    batched_embeddings = [embeddings[i:i+batch_size] for i in range(0, len(embeddings), batch_size)]

    all_top_k_labels = []
    all_top_k_similarities = []

    from tqdm import tqdm
    for batch_embeddings in tqdm(batched_embeddings):
        
        if is_host_host:
            batch_top_k_labels, batch_top_k_similarities = similarity_dataset.get_host_host_cosine_similarity(batch_embeddings, top_k)
        else:
            batch_top_k_labels, batch_top_k_similarities = similarity_dataset.get_host_cosine_similarity(batch_embeddings, top_k)
        
        all_top_k_labels.extend(batch_top_k_labels)
        all_top_k_similarities.extend(batch_top_k_similarities)

    return np.array(all_top_k_labels), np.array(all_top_k_similarities)


def get_next_sentence_pairs(document_path, ip_documents):    
    labelled_sentence_pairs = []
    ip1_list = []
    ip2_list = []
    file_list = os.listdir(document_path)
    for file in file_list:
        with open(document_path + "/" + file, 'r') as f:
            ip1 = file.split(".txt")[0]
            if ip1 in ip_documents:
                position = ip_documents[ip1]["position"]
                f.seek(position)
                lines = f.readlines()
                ip_documents[ip1]["position"] = f.tell()
            else:
                lines = f.readlines()
                ip_documents[ip1] = {"position": f.tell()}
                        
            for line in lines:
                line.strip()
                ip2, start_time, end_time, sentence = line.split(",")
                last_sentence = None
                if ip2 in ip_documents[ip1]:
                    last_sentence = ip_documents[ip1][ip2]
                    ip_documents[ip1][ip2] = sentence
                    labelled_sentence_pairs.append((last_sentence, sentence))
                    ip1_list.append(ip1)
                    ip2_list.append(ip2)
                else:
                    ip_documents[ip1][ip2] = sentence # Don't use single sentence documents for classification
    return ip1_list, ip2_list, labelled_sentence_pairs
