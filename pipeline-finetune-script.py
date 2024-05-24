validation_strategies = [] # TODO: add different strategies, i.e., how much context for classification
validation_year = 2022

import json
pretrained_model_path = 'exp3_models/model_0.417_1-aug_normal_1'
experiment_name = 'exp3 fine aug - ' + pretrained_model_path.split('-')[-1]
with open(pretrained_model_path + '/hyperparameters.json') as f:
    hyperparameters = json.load(f)

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

train_data = BERTClassDataset(train_corpus, train_labels, seq_length=hyperparameters['seq_length'], voc_min=5, voc_max=num_buckets + 4, augment=True, shuffle=True, max_shift_amt=2)
eval_data = BERTClassDataset(validation_corpus, validation_labels, seq_length=hyperparameters['seq_length'], augment=False)


from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=2)

import os
finetune_model_path = "checkpoints_finetuning"
folders = os.listdir(finetune_model_path)
ids = [int(folder.split('_')[-1]) for folder in folders]
max_id = max(ids) if ids else 0
finetune_model_path = os.path.join(finetune_model_path, f"model_{max_id + 1}")
try:
    os.mkdir(finetune_model_path)
except:
    pass


training_args = TrainingArguments(
    output_dir=finetune_model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=3,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=96, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=256,  # evaluation batch size
    logging_steps=500,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=500,
    greater_is_better=False,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    save_total_limit=3,
    run_name=experiment_name,
    # do_eval=False,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
)

trainer.train()


result = trainer.evaluate()
print("Results: ", result)

import os

models_path = "exp3_finetune"

# list all folders in models
folders = os.listdir(models_path)

# extract id in each folder
ids = [int(folder.split('_')[-1]) for folder in folders]

# get the maximum id
max_id = max(ids) if ids else 0

# get the evaluation loss
eval_loss = result['eval_loss']

# format eval_loss to first 3 decimal places
eval_loss = "{:.3f}".format(eval_loss)

dir_name = f"model_{eval_loss}_{max_id + 1}"
path = os.path.join(models_path, dir_name)
if not os.path.exists(path):
    os.makedirs(path)

# save the model
trainer.save_model(path)

print(f"Model saved to {path}")

print(hyperparameters)
hyperparameters['result'] = result


# save hyperparameters to the model directory
with open(os.path.join(path, 'hyperparameters.json'), 'w') as f:
    json.dump(hyperparameters, f)