sentence_types = ['f', 'i']
word_types = ['b', 'p']
hp_filter_sizes = [0, 100, 150, 50]
burst_timeouts = [1000, 30000, 50000]
sentence_timeouts = [500000, 1000000, 15000000]
labeling_granularity = ['sentence', 'document', 'host']
ngram_sizes = [2, 3, 4, 5]
buckets = [5, 20, 50, 30]
percentiles = [0.95, 0.99]
validation_year = 2022
training_years = [2017, 2018, 2019, 2020, 2021, 2022]
token_features = ['log_bytes', 'avg_bytes']
log_maxs = [15, 16]
avg_maxs = [1500, 2500]
seq_lengths = [128, 512, 64]
doc_lengths = [100, 50]
embedding_lengths = [128, 256, 768, 512]
num_attention_heads = [4, 8, 12, 10]
num_hidden_layers = [4, 8, 12, 10]

sentence_type_id = 0
word_type_id = 1
hp_filter_size_id = 1
burst_timeout_id = 0
sentence_timeout_id = 1
labeling_granularity_id = 1
ngram_size_id = 1
bucket_id = 1
percentile_id = 0
validation_year_id = 0
training_year_ids = [1, 2, 3, 4, 5]
token_feature_id = 0
seq_length_id = 0
doc_length_id = 1
max_id = 1
embedding_length_id = 1
num_attention_heads_id = 1
num_hidden_layers_id = 1

filter_singletons = True
do_training = True
do_validation1 = True
do_validation2 = True
do_validation3 = False

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
            if (feature_name != token_features[token_feature_id]):
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

if word_types[word_type_id] == 'p':
    burst_timeout = 'x'
else:
    burst_timeout = str(burst_timeouts[burst_timeout_id])

for year_id in training_year_ids:
    print('Reading year', training_years[year_id])
    path = 'config_' + sentence_types[sentence_type_id] + word_types[word_type_id] + '_' + str(hp_filter_sizes[hp_filter_size_id]) + '_' + burst_timeout + '_' + str(sentence_timeouts[sentence_timeout_id]) + '.zip'
    contents, names, ips = read_zip_archive(str(training_years[year_id]) + '/' + path, contents, names, ips)


import numpy as np

num_buckets = buckets[bucket_id]
min_buckets = 0
if token_features[token_feature_id] == 'log_bytes':
    max_buckets = log_maxs[max_id]
else:
    max_buckets = avg_maxs[max_id]

# Define the bucket boundaries
bucket_boundaries = np.linspace(min_buckets, max_buckets, num_buckets)

def bucket_id_from_decimal(decimal):
    # Use np.digitize to find the bucket that each decimal belongs to
    # 0 - 4 is reserved for special tokens
    return np.digitize(decimal, bucket_boundaries) + 4

corpus = []
for content in tqdm.tqdm(contents):
    document = []
    corpus.append(document)
    for sentence_id, row in enumerate(content.split('\n')):
        if sentence_id > doc_lengths[doc_length_id]:
            break
        sentence = []
        for i, value in enumerate(row.split(' ')):
            if i > seq_lengths[seq_length_id]:
                break
            if value == '':
                continue
            value = float(value)
            sentence.append(bucket_id_from_decimal(value))
        document.append(sentence)
        
        
deduplicated_corpus = []
for document in corpus:
    document = document[:100]
    deduplicated_corpus.append(document)
    
    
    
from sklearn.model_selection import train_test_split
import numpy as np

np.random.shuffle(deduplicated_corpus)
train_corpus, validation_corpus = train_test_split(deduplicated_corpus, test_size=0.01, random_state=42)



from tqdm import tqdm
from bert_dataset import BERTDataset
from torch.utils.data import DataLoader

if do_training:
    train_data = BERTDataset(train_corpus, seq_length=seq_lengths[seq_length_id], min_bucket=5, max_bucket=4 + num_buckets, augment=False)
    # train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=False)
    eval_data = BERTDataset(validation_corpus, seq_length=seq_lengths[seq_length_id], min_bucket=5, max_bucket=4 + num_buckets, augment=False)
    # eval_loader = DataLoader(eval_data, batch_size=32, shuffle=False, pin_memory=False)
    
    

from transformers import BertConfig
from transformers import BertForPreTraining

model_config = BertConfig(vocab_size=num_buckets + 5, 
                          max_position_embeddings=seq_lengths[seq_length_id],
                          hidden_size=embedding_lengths[embedding_length_id],
                          intermediate_size=4 * embedding_lengths[embedding_length_id],
                          num_hidden_layers=8,
                          num_attention_heads=8,)
model = BertForPreTraining(config=model_config)



from transformers import Trainer, TrainingArguments

import os
model_path = "checkpoints"
folders = os.listdir(model_path)
ids = [int(folder.split('_')[-1]) for folder in folders]
max_id = max(ids) if ids else 0
model_path = os.path.join(model_path, f"model_{max_id + 1}")
try:
    os.mkdir(model_path)
except:
    pass

training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=3,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=256, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=256,  # evaluation batch size
    logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,
    greater_is_better=False,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    save_total_limit=3,
    
    run_name='exp4_fp-all-years-med-timeout',
    
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


import os

models_path = "exp3_models"

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

import json
# Assuming the hyperparameters are stored in a dictionary named `hyperparameters`
hyperparameters = {
    'sentence_type': sentence_types[sentence_type_id],
    'word_type': word_types[word_type_id],
    'hp_filter_size': hp_filter_sizes[hp_filter_size_id],
    'burst_timeout': burst_timeouts[burst_timeout_id],
    'sentence_timeout': sentence_timeouts[sentence_timeout_id],
    'num_buckets': buckets[bucket_id],
    'bucket_min': min_buckets,
    'bucket_max': max_buckets,
    'bucket_boundaries': bucket_boundaries.tolist(),
    'seq_length': seq_lengths[seq_length_id],
    'doc_length': doc_lengths[doc_length_id],
    'embedding_length': embedding_lengths[embedding_length_id],
    'train_batch_size': 32,
    'eval_batch_size': 32,
    'vocab_size': num_buckets + 5,
    'max_position_embeddings': seq_lengths[seq_length_id],
    'hidden_size': embedding_lengths[embedding_length_id],
    'intermediate_size': 4 * embedding_lengths[embedding_length_id],
    'num_hidden_layers': num_hidden_layers[num_hidden_layers_id],
    'num_attention_heads': num_attention_heads[num_attention_heads_id],
    'years': [training_years[year_id] for year_id in training_year_ids],
    'feature': token_features[token_feature_id],
    'num_train_epochs': 3,
    'per_device_train_batch_size': 32,
    'gradient_accumulation_steps': 8,
    'per_device_eval_batch_size': 128,
    'logging_steps': 1000,
    'save_steps': 1000,
    'greater_is_better': False,
    'metric_for_best_model': "eval_loss",
    'load_best_model_at_end': True,
    'save_total_limit': 3
}

hyperparameters['evaluation_result'] = result

# Write to json file
with open(os.path.join(path, 'hyperparameters.json'), 'w') as f:
    json.dump(hyperparameters, f, indent=4)

print('Model saved to', path)
print('Hyperparameters:', hyperparameters)