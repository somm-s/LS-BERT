import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

path = "document_datasets"

class SimilarityDataset():
    def __init__(self, year, fine=True, label_path=None, use_unknown=False, only_rt=True, max_host_host_samples=None, max_host_samples=None):
        self.year = year
        self.fine = fine
        self.label_path = label_path
        self.use_unknown = use_unknown
        self.max_host_host_samples = max_host_host_samples
        self.max_host_samples = max_host_samples
        self.only_rt = only_rt
        self.dataset = self.load_dataset()

    def load_dataset(self):
        suffix = "_pair_fine" if self.fine else "_pair_pre"
        load_path = "document_datasets/" + str(self.year) + suffix

        self.host_embeddings = np.load(load_path + '/host_embeddings.npy')
        self.host_host_embeddings = np.load(load_path + '/host_host_embeddings.npy')
        with open(load_path + '/metadata_corpus.json', 'r') as f:
            self.metadata_corpus = json.load(f)
        with open(load_path + '/hosts.txt', 'r') as f:
            self.hosts = f.read().splitlines()
        with open(load_path + '/host_hosts.txt', 'r') as f:
            self.host_hosts = f.read().splitlines()
        self.host_hosts = [tuple(x.split(',')) for x in self.host_hosts]
        
        
        # filter out broken embeddings (nan arrays or arrays with all zeros or with pos/neg inf)
        valid_host_indices = []
        for i, emb in enumerate(self.host_embeddings):
            if np.isnan(emb).any() or np.isinf(emb).any() or np.all(emb == 0):
                continue
            valid_host_indices.append(i)
        self.host_embeddings = self.host_embeddings[valid_host_indices]
        self.hosts = [self.hosts[i] for i in valid_host_indices]
        
        valid_host_host_indices = []
        for i, emb in enumerate(self.host_host_embeddings):
            if np.isnan(emb).any() or np.isinf(emb).any() or np.all(emb == 0):
                continue
            valid_host_host_indices.append(i)
        self.host_host_embeddings = self.host_host_embeddings[valid_host_host_indices]
        self.host_hosts = [self.host_hosts[i] for i in valid_host_host_indices]
        
        self.host_host_labels = [0] * len(self.host_hosts)
        self.host_labels = [0] * len(self.hosts)
        
        if self.label_path is not None:
            with open(self.label_path, 'r') as f:
                labels = f.read().splitlines()
            labels = {x.split(',')[0]: int(x.split(',')[1]) for x in labels}
            
            for i, (ip1, ip2) in enumerate(self.host_hosts):
                label_1_val = 0
                label_2_val = 0
                if ip1 in labels:
                    label_1_val = labels[ip1]
                if ip2 in labels:
                    label_2_val = labels[ip2]
                self.host_host_labels[i] = max(label_1_val, label_2_val)
            for i, ip in enumerate(self.hosts):
                if ip in labels:
                    self.host_labels[i] = labels[ip]
        
        if not self.use_unknown:
            self.host_host_embeddings = [x for i, x in enumerate(self.host_host_embeddings) if self.host_host_labels[i] > 1]
            self.host_hosts = [x for i, x in enumerate(self.host_hosts) if self.host_host_labels[i] > 1]
            self.host_host_labels = [x for x in self.host_host_labels if x > 1]
            
            self.host_embeddings = [x for i, x in enumerate(self.host_embeddings) if self.host_labels[i] > 1]
            self.hosts = [x for i, x in enumerate(self.hosts) if self.host_labels[i] > 1]
            self.host_labels = [x for x in self.host_labels if x > 1]
        
        if self.only_rt:
            self.host_host_embeddings = [x for i, x in enumerate(self.host_host_embeddings) if self.host_host_labels[i] == 2]
            self.host_hosts = [x for i, x in enumerate(self.host_hosts) if self.host_host_labels[i] == 2]
            self.host_host_labels = [x for x in self.host_host_labels if x == 2]
            
            self.host_embeddings = [x for i, x in enumerate(self.host_embeddings) if self.host_labels[i] == 2]
            self.hosts = [x for i, x in enumerate(self.hosts) if self.host_labels[i] == 2]
            self.host_labels = [x for x in self.host_labels if x == 2]
        
        if self.max_host_host_samples is not None and len(self.host_host_embeddings) > self.max_host_host_samples:
            indices = np.random.choice(len(self.host_hosts), self.max_host_host_samples, replace=False)
            self.host_host_embeddings = [self.host_host_embeddings[i] for i in indices]
            self.host_hosts = [self.host_hosts[i] for i in indices]
            self.host_host_labels = [self.host_host_labels[i] for i in indices]
            
        if self.max_host_samples is not None and len(self.host_embeddings) > self.max_host_samples:
            indices = np.random.choice(len(self.hosts), self.max_host_samples, replace=False)
            self.host_embeddings = [self.host_embeddings[i] for i in indices]
            self.hosts = [self.hosts[i] for i in indices]
            self.host_labels = [self.host_labels[i] for i in indices]
            
        self.host_embeddings = np.array(self.host_embeddings)
        self.host_host_embeddings = np.array(self.host_host_embeddings)
        self.host_labels = np.array(self.host_labels)
        self.host_host_labels = np.array(self.host_host_labels)
        self.hosts = np.array(self.hosts)
        self.host_hosts = np.array(self.host_hosts)

    def get_host_cosine_similarity(self, embeddings, k):
        similarities = cosine_similarity(embeddings, self.host_embeddings)
        top_k = np.argsort(-similarities, axis=1)[:, :k]
        top_k_labels = np.array([self.host_labels[i] for i in top_k.flatten()], dtype=int).reshape(top_k.shape)
        top_k_similarities = np.array([similarities[i][top_k[i]] for i in range(len(embeddings))])    
        return top_k_labels, top_k_similarities
    
    def get_host_host_cosine_similarity(self, embeddings, k):
        similarities = cosine_similarity(embeddings, self.host_host_embeddings)
        top_k = np.argsort(-similarities, axis=1)[:, :k]
        top_k_labels = np.array([self.host_host_labels[i] for i in top_k.flatten()], dtype=int).reshape(top_k.shape)
        top_k_similarities = np.array([similarities[i][top_k[i]] for i in range(len(embeddings))])    
        return top_k_labels, top_k_similarities
    
    def get_rt_host_embeddings(self):
        return self.host_embeddings[self.host_labels == 2], self.hosts[self.host_labels == 2]
    
    def get_benign_host_embeddings(self):
        return self.host_embeddings[self.host_labels == 3], self.hosts[self.host_labels == 3]
    
    def get_rt_host_host_embeddings(self):
        return self.host_host_embeddings[self.host_host_labels == 2], self.host_hosts[self.host_host_labels == 2]
    
    def get_benign_host_host_embeddings(self):
        return self.host_host_embeddings[self.host_host_labels == 3], self.host_hosts[self.host_host_labels == 3]
    