finetuned_model_path = "exp1_finetune/model_0.187_9"

evaluation_name = finetuned_model_path.replace('/', '_')
output_path = "model_evaluations/" + evaluation_name
import os
if not os.path.exists(output_path):
    os.makedirs(output_path)

validation_year = 2022
validation_2020 = True

config = finetuned_model_path + "/config.json"
import json
with open(config, 'r') as f:
    config = json.load(f)
    pretrained_model_path = config['_name_or_path']
    
print("Pretrained model path: ", pretrained_model_path)
with open(pretrained_model_path + '/hyperparameters.json') as f:
    hyperparameters = json.load(f)
    
    
    
################## Loading datasets ##################
import zipfile
import tqdm

def read_zip_archive(zip_file_path, contents, names, ips):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        for file_name in tqdm.tqdm(file_list):
            if (not file_name.endswith('.csv')):
                continue
            name = file_name.split('/')[-1]
            feature_name = file_name.split('/')[-2]
            if (feature_name != hyperparameters["feature"]):
                continue
            name = name.split('.csv')[0]
            ip1 = name.split('-')[0]
            ip2 = name.split('-')[1]
            names.append(name)
            ips.add(ip1)
            ips.add(ip2)
            # Read the content of each file into a list of strings
            with zip_ref.open(file_name) as file:
                content = file.read().decode('utf-8')  # Assuming the content is in UTF-8 encoding
                contents.append(content)
    return contents, names, ips


contents = []
names = []
ips = set()

burst_timeout = 'x' if hyperparameters['word_type'] == 'p' else str(hyperparameters['burst_timeout'])

print('Reading year 2022')
path = 'config_' + hyperparameters['sentence_type'] + hyperparameters['word_type'] + '_' + str(hyperparameters["hp_filter_size"]) + '_' + burst_timeout + '_' + str(hyperparameters["sentence_timeout"]) + '.zip'
contents, names, ips = read_zip_archive(str(validation_year) + '/' + path, contents, names, ips)

import pandas as pd
ip_labels = pd.read_csv('ip_labels.txt', header=None, names=['ip', 'label'])
ip_label_dict = ip_labels.set_index('ip')['label'].to_dict()


teamserver_1 = [
    "37.157.123.51",
    "37.157.127.17",
    "2a00:16e0:100:0:7abd:2745:d741:c8a0",
    "2001:1bf1:0:0:9256:f3bb:eb24:fb84",
    "185.20.58.178",
    "46.131.46.67",
    "2a04:80c2:0:0:738b:3848:25e4:800c",
    "2a07:1182:1000:1002:19a7:df71:7fe7:6522",
    "94.246.252.193",
]

if validation_2020:
    contents_2020 = []
    names_2020 = []
    ips_2020 = set()
    contents_2020, names_2020, ips_2020 = read_zip_archive(str(2020) + '/' + path, contents_2020, names_2020, ips_2020)

    teamserver_documents = []
    for content, name in zip(contents_2020, names_2020):
        if any(ip in name for ip in teamserver_1):
            teamserver_documents.append(content)
        
    benign_2020 = []
    for ip in ips_2020:
        if ip in ips and ip_label_dict[ip] == 3:
            benign_2020.append(ip)
    benign_2020_documents = []
    for content, name in zip(contents_2020, names_2020):
        if any(ip in name for ip in benign_2020):
            benign_2020_documents.append(content)
            
            
import numpy as np

num_buckets = hyperparameters["num_buckets"]
min_buckets = hyperparameters["bucket_min"]
max_buckets = hyperparameters["bucket_max"]
bucket_boundaries = np.linspace(min_buckets, max_buckets, num_buckets)

def bucket_id_from_decimal(decimal):
    return np.digitize(decimal, bucket_boundaries) + 4 # five reserved tokens, 0-4, digitize starts at 1

corpus = []
for content in tqdm.tqdm(contents):
    document = []
    corpus.append(document)
    for sentence_id, row in enumerate(content.split('\n')):
        if sentence_id > hyperparameters['doc_length']:
            break
        sentence = []
        for i, value in enumerate(row.split(' ')):
            if i > hyperparameters['seq_length']:
                break
            if value == '':
                continue
            value = float(value)
            sentence.append(bucket_id_from_decimal(value))
        document.append(sentence)

if validation_2020:
    teamserver_corpus = []
    for content in tqdm.tqdm(teamserver_documents):
        document = []
        teamserver_corpus.append(document)
        for sentence_id, row in enumerate(content.split('\n')):
            if sentence_id > hyperparameters['doc_length']:
                break
            sentence = []
            for i, value in enumerate(row.split(' ')):
                if i > hyperparameters['seq_length']:
                    break
                if value == '':
                    continue
                value = float(value)
                sentence.append(bucket_id_from_decimal(value))
            document.append(sentence)
            
    benign_2020_corpus = []
    for content in tqdm.tqdm(benign_2020_documents):
        document = []
        benign_2020_corpus.append(document)
        for sentence_id, row in enumerate(content.split('\n')):
            if sentence_id > hyperparameters['doc_length']:
                break
            sentence = []
            for i, value in enumerate(row.split(' ')):
                if i > hyperparameters['seq_length']:
                    break
                if value == '':
                    continue
                value = float(value)
                sentence.append(bucket_id_from_decimal(value))
            document.append(sentence)
            
            
# 0: internal, 1: firewall, 2: rt, 3: other
# only allow rt-internal and rt-firewall as rt (label 0)
# use internal-other and firewall-other as other (label 1)
def label_rule(label1, label2):
    if label1 == 0 and label2 == 2 or label2 == 0 and label1 == 2 or label1 == 1 and label2 == 2 or label2 == 1 and label1 == 2:
        return 0
    if label1 == 0 and label2 == 3 or label2 == 0 and label1 == 3 or label1 == 1 and label2 == 3 or label2 == 1 and label1 == 3:
        return 1
    return 2

cleaned_corpus = []
labels = []
for name, document in zip(names, corpus):
    ip1 = name.split('-')[0]
    ip2 = name.split('-')[1]
    label1 = ip_label_dict[ip1]
    label2 = ip_label_dict[ip2]
    label = label_rule(label1, label2)
    if label == 2:
        continue
    labels.append(label)
    cleaned_corpus.append(document)
    
from sklearn.model_selection import train_test_split
import numpy as np


combined = list(zip(cleaned_corpus, labels))
np.random.shuffle(combined)
cleaned_corpus_shuffled, labels_shuffled = zip(*combined)

train_corpus, validation_corpus, train_labels, validation_labels = train_test_split(cleaned_corpus_shuffled, labels_shuffled, test_size=0.2, random_state=42, stratify=labels_shuffled)


from bert_class_dataset import BERTClassDataset

train_data = BERTClassDataset(train_corpus, train_labels, seq_length=hyperparameters['seq_length'], augment=False)
eval_data = BERTClassDataset(validation_corpus, validation_labels, seq_length=hyperparameters['seq_length'], augment=False)

if validation_2020:
    teamserver_data = BERTClassDataset(teamserver_corpus, [0] * len(teamserver_corpus), seq_length=hyperparameters['seq_length'], augment=False)
    benign_2020_data = BERTClassDataset(benign_2020_corpus, [1] * len(benign_2020_corpus), seq_length=hyperparameters['seq_length'], augment=False)
    
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(finetuned_model_path)
model.to('cuda')

from torch.utils.data import DataLoader
eval_loader = DataLoader(eval_data, batch_size=64, shuffle=True, pin_memory=False)

model.eval()
all_outputs = []
all_labels = []
# iterate over the batches
for batch in tqdm.tqdm(eval_loader):
    # move the batch to the device
    labels = batch['label']
    all_labels.append(labels)
    batch = {k: v.to('cuda') for k, v in batch.items() if k != 'label'}
    # forward pass
    outputs = model.base_model(**batch)
    all_outputs.append(outputs.last_hidden_state[:,0,:].cpu().detach().numpy())
all_outputs = np.concatenate(all_outputs, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

if validation_2020:
    teamserver_loader = DataLoader(teamserver_data, batch_size=64, shuffle=True, pin_memory=False)
    benign_2020_loader = DataLoader(benign_2020_data, batch_size=64, shuffle=True, pin_memory=False)
    model.eval()
    teamserver_outputs = []
    benign_2020_outputs = []
    # iterate over the batches
    for batch in tqdm.tqdm(teamserver_loader):
        batch = {k: v.to('cuda') for k, v in batch.items() if k != 'label'}
        outputs = model.base_model(**batch)
        teamserver_outputs.append(outputs.last_hidden_state[:,0,:].cpu().detach().numpy())
    teamserver_outputs = np.concatenate(teamserver_outputs, axis=0)
    for i, batch in tqdm.tqdm(enumerate(benign_2020_loader)):
        if i > 50:
            break
        batch = {k: v.to('cuda') for k, v in batch.items() if k != 'label'}
        outputs = model.base_model(**batch)
        benign_2020_outputs.append(outputs.last_hidden_state[:,0,:].cpu().detach().numpy())
    benign_2020_outputs = np.concatenate(benign_2020_outputs, axis=0)

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

np.random.seed(42)
indices = np.random.choice(all_outputs.shape[0], 10000, replace=False)
embeddings_sample = all_outputs[indices]
embeddings_labels = all_labels[indices]

if validation_2020:
    embeddings_sample = np.concatenate([embeddings_sample, teamserver_outputs, benign_2020_outputs], axis=0)
    embeddings_labels = np.concatenate([embeddings_labels, [2] * len(teamserver_outputs), [3] * len(benign_2020_outputs)], axis=0)
    
    
legend = ['RT 2022', 'Benign 2022', 'RT Validation', 'Benign Validation']

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings_sample)

# Plot the reduced representations
plt.figure(figsize=(10, 8))
# plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=embeddings_labels, alpha=0.7, s=10)

for i, label in enumerate(legend):
    indices = embeddings_labels == i
    plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], alpha=0.7, s=10, label=label)


plt.title('Last Hidden Layer Representations after Fine-Tuning')
plt.legend()
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.savefig(output_path + '/tsne_finetuning.png')


experiment_results = {}
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


####################### Fine-Tune embeddings with RF and SVM on trained on test data 2022 and validated on 2020 #######################
np.random.seed(43)
indices = np.random.choice(all_outputs.shape[0], 5000, replace=False)
X = all_outputs[indices]
y = all_labels[indices]

clf = RandomForestClassifier(n_estimators=100, random_state=43)
clf.fit(X, y)

clf_svm = SVC(random_state=43)
clf_svm.fit(X, y)

X_val = np.concatenate([teamserver_outputs, benign_2020_outputs], axis=0)
y_val = np.concatenate([np.zeros(len(teamserver_outputs)), np.ones(len(benign_2020_outputs))], axis=0)

# shuffle
indices = np.arange(X_val.shape[0])
np.random.shuffle(indices)
X_val = X_val[indices]
y_val = y_val[indices]

# Predict on validation data
valid_preds = clf.predict(X_val)
valid_preds_svm = clf_svm.predict(X_val)
from sklearn.metrics import classification_report
experiment_results["rf_validation_finetune"] = classification_report(y_val, valid_preds, target_names=['RT', 'Benign'])
experiment_results["svm_validation_finetune"] = classification_report(y_val, valid_preds_svm, target_names=['RT', 'Benign'])

from torch.utils.data import DataLoader
eval_loader = DataLoader(eval_data, batch_size=64, shuffle=False, pin_memory=False)

model.eval()
all_true_labels = []
all_logits = []
all_pred_labels = []
# iterate over the batches
for batch in tqdm.tqdm(eval_loader):
    # move the batch to the device
    labels = batch['label']
    all_true_labels.append(labels)
    batch = {k: v.to('cuda') for k, v in batch.items() if k != 'label'}
    # forward pass
    outputs = model(**batch)
    all_logits.append(outputs.logits.cpu().detach().numpy())
    
if validation_2020:
    teamserver_loader = DataLoader(teamserver_data, batch_size=64, shuffle=False, pin_memory=False)
    benign_2020_loader = DataLoader(benign_2020_data, batch_size=64, shuffle=False, pin_memory=False)
    
    model.eval()
    teamserver_logits = []
    # iterate over the batches
    for batch in tqdm.tqdm(teamserver_loader):
        batch = {k: v.to('cuda') for k, v in batch.items() if k != 'label'}
        outputs = model(**batch)
        teamserver_logits.append(outputs.logits.cpu().detach().numpy())
    
    benign_2020_logits = []
    # iterate over the batches
    i = 0
    for batch in tqdm.tqdm(benign_2020_loader):
        if i > 50:
            break
        i += 1
        batch = {k: v.to('cuda') for k, v in batch.items() if k != 'label'}
        outputs = model(**batch)
        benign_2020_logits.append(outputs.logits.cpu().detach().numpy())
    
    benign_2020_logits = np.concatenate(benign_2020_logits, axis=0)
    teamserver_logits = np.concatenate(teamserver_logits, axis=0)
    teamserver_true_labels = [0] * len(teamserver_logits)
    benign_2020_true_labels = [1] * len(benign_2020_logits)
    teamserver_pred_labels = np.argmax(teamserver_logits, axis=1)
    benign_2020_pred_labels = np.argmax(benign_2020_logits, axis=1)
    all_2020_true_labels = np.concatenate([teamserver_true_labels, benign_2020_true_labels], axis=0)
    all_2020_pred_labels = np.concatenate([teamserver_pred_labels, benign_2020_pred_labels], axis=0)
    valid_class_finetune_result = classification_report(all_2020_true_labels, all_2020_pred_labels, target_names=['RT', 'Benign'])
    
    experiment_results['validation_finetune_model'] = valid_class_finetune_result
    

all_logits = np.concatenate(all_logits, axis=0)
all_true_labels = np.concatenate(all_true_labels, axis=0)

# convert logits to labels
all_pred_labels = np.argmax(all_logits, axis=1)

import torch

softmax = torch.nn.Softmax(dim=1)
all_logits_softmax = softmax(torch.Tensor(all_logits))
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(all_true_labels, all_logits_softmax[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# save finetune-roc to output path
plt.savefig(output_path + '/roc_finetuning.png')

experiment_results["test_finetune_model"] = classification_report(all_true_labels, all_pred_labels)

with open(output_path + '/fine-tune-results.txt', 'w') as f:
    for key in experiment_results:
        f.write(key + '\n')
        f.write(experiment_results[key] + '\n')
        f.write('\n')











################ Pretrained Embeddings For Classification ################
from transformers import AutoModel
model = AutoModel.from_pretrained(pretrained_model_path)
model.to('cuda')
model.eval()

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Create DataLoader for training and validation data
train_loader = DataLoader(train_data, batch_size=128, shuffle=False, pin_memory=False)
valid_loader = DataLoader(eval_data, batch_size=128, shuffle=False, pin_memory=False)

# Function to create embeddings
def create_embeddings(loader):
    model.eval()
    embeddings = []
    labels = []
    for batch in tqdm.tqdm(loader):
        labels.extend(batch['label'].numpy())
        batch = {k: v.to('cuda') for k, v in batch.items() if k != 'label'}
        with torch.no_grad():
            outputs = model.base_model(**batch)
        embeddings.extend(outputs.last_hidden_state[:,0,:].cpu().numpy())
    return np.array(embeddings), np.array(labels)

# Create embeddings for training and validation data
train_embeddings, train_labels = create_embeddings(train_loader)
valid_embeddings, valid_labels = create_embeddings(valid_loader)


if validation_2020:
    teamserver_loader = DataLoader(teamserver_data, batch_size=64, shuffle=True, pin_memory=False)
    benign_2020_loader = DataLoader(benign_2020_data, batch_size=64, shuffle=True, pin_memory=False)
    
    model.eval()
    teamserver_outputs = []
    # iterate over the batches
    for batch in tqdm.tqdm(teamserver_loader):
        batch = {k: v.to('cuda') for k, v in batch.items() if k != 'label'}
        with torch.no_grad():
            outputs = model.base_model(**batch)
        teamserver_outputs.append(outputs.last_hidden_state[:,0,:].cpu().detach().numpy())
    teamserver_outputs = np.concatenate(teamserver_outputs, axis=0)
    
    benign_2020_outputs = []
    # iterate over the batches
    i = 0
    for batch in tqdm.tqdm(benign_2020_loader):
        if i > 50:
            break
        i += 1
        batch = {k: v.to('cuda') for k, v in batch.items() if k != 'label'}
        with torch.no_grad():
            outputs = model.base_model(**batch)
        benign_2020_outputs.append(outputs.last_hidden_state[:,0,:].cpu().detach().numpy())
    benign_2020_outputs = np.concatenate(benign_2020_outputs, axis=0)
    
    
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

np.random.seed(42)
indices = np.random.choice(valid_embeddings.shape[0], 10000, replace=False)
embeddings_sample = valid_embeddings[indices]
embeddings_labels = valid_labels[indices]

if validation_2020:
    embeddings_sample = np.concatenate([embeddings_sample, teamserver_outputs, benign_2020_outputs], axis=0)
    embeddings_labels = np.concatenate([embeddings_labels, [2] * len(teamserver_outputs), [3] * len(benign_2020_outputs)], axis=0)
    
legend = ['RT 2022', 'Benign 2022', 'RT Validation', 'Benign Validation']

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings_sample)

# Plot the reduced representations
plt.figure(figsize=(10, 8))
# plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=embeddings_labels, alpha=0.7, s=10)

for i, label in enumerate(legend):
    indices = embeddings_labels == i
    plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], alpha=0.7, s=10, label=label)


plt.title('Last Hidden Layer Representations after Pre-Training')
plt.legend()
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# save to output path
plt.savefig(output_path + '/tsne_pretraining.png')

from sklearn.utils import resample

# Use train_embeddings as X and train_labels as y
X = train_embeddings
y = train_labels
n_samples_per_class = 5000 # min(np.bincount(y))

# Separate majority and minority classes
X_0 = X[y == 0]
X_1 = X[y == 1]

# Downsample majority class and upsample minority class
X_0_downsampled = resample(X_0, replace=False, n_samples=n_samples_per_class, random_state=42)
X_1_downsampled = resample(X_1, replace=False, n_samples=n_samples_per_class, random_state=42)

# Combine downsampled majority class with minority class
X_downsampled = np.concatenate([X_0_downsampled, X_1_downsampled], axis=0)

# Create corresponding y labels
y_downsampled = np.array([0]*n_samples_per_class + [1]*n_samples_per_class)

# Use X_downsampled and y_downsampled for training
X = X_downsampled
y = y_downsampled

# shuffle
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

clf_svm = SVC(random_state=42)
clf_svm.fit(X, y)

# Predict on validation data
valid_preds = clf.predict(valid_embeddings)
valid_preds_svm = clf_svm.predict(valid_embeddings)

experiment_results = {}

experiment_results["rf_test_pretrained"] = classification_report(valid_labels, valid_preds, target_names=['RT', 'Benign'])
experiment_results["svm_test_pretrained"] = classification_report(valid_labels, valid_preds_svm, target_names=['RT', 'Benign'])

teamserver_preds = clf.predict(teamserver_outputs)
benign_preds_2020 = clf.predict(benign_2020_outputs)
all_preds_2020 = np.concatenate([teamserver_preds, benign_preds_2020], axis=0)
all_labels_2020 = np.concatenate([np.zeros(len(teamserver_preds)), np.ones(len(benign_preds_2020))], axis=0)

# svm
teamserver_preds = clf_svm.predict(teamserver_outputs)
benign_preds_2020 = clf_svm.predict(benign_2020_outputs)
all_preds_2020_svm = np.concatenate([teamserver_preds, benign_preds_2020], axis=0)

experiment_results["rf_validation_pretrained"] = classification_report(all_labels_2020, all_preds_2020, target_names=['RT', 'Benign'])
experiment_results["svm_validation_pretrained"] = classification_report(all_labels_2020, all_preds_2020_svm, target_names=['RT', 'Benign'])

with open(output_path + '/pre-train-results.txt', 'w') as f:
    for key in experiment_results:
        f.write(key + '\n')
        f.write(experiment_results[key] + '\n')
        f.write('\n')