# configuration = "fi-100_ft-15000000_nb-30_ht-100000_di-uni_mx-11_lg-yes"
# configuration = "fi-100_ft-15000000_nb-30_ht-100000_di-bi_mx-1500_lg-no"
# configuration = "fi-100_ft-15000000_nb-30_ht-0_di-uni_mx-1500_lg-no"
configuration = "fi-100_ft-15000000_nb-30_ht-100000_di-uni_mx-1500_lg-no"
# configuration = "fi-100_ft-15000000_nb-20_ht-100000_di-uni_mx-1500_lg-no"

dataset_path = "datasets"
pre_train_years = [2017, 2018, 2019, 2020]
train_year = 2022
validation_year = 2021
output_dir = "trained_models"

#### Pre-Processing:
seq_length = 128
deduplicate = True
coalesce = True

#### Pre-Training:
do_pre_train = False
pre_train_run_name = "pre_train_all_years_ded_coa_med_seq-128_abl_" + configuration
# for largest configuration: use batch size of 48 on A100, 16 on others
# medium configuration: 128 on others
per_device_batch_size = 256
embedding_length = 256
num_attention_heads = 12
num_hidden_layers = 12
num_epochs = 2

#### Fine-Tuning:
do_fine_tune = False
fine_tune_run_name = "fine_tune_all_years_ded_coa_med_seq-128_abl_" + configuration
# fine_tune_run_name = "fine_tune_all_years_ded_coa_med_seq-128_abl_" + configuration

num_fine_tune_epochs = 1
fine_tune_per_device_batch_size = 128
fine_tune_per_device_eval_batch_size = 256

#### Evaluation:
do_evaluate = True
use_finetuned_model = True

from data_loader import *
if do_pre_train:
    pre_train_corpus = load_pretrain_data(pre_train_years, configuration, dataset_path)
if do_fine_tune or do_evaluate:
    train_benign_corpus, train_rt_corpus = load_train_data(train_year, configuration, dataset_path)
    valid_benign_corpus, valid_rt_corpus = load_train_data(validation_year, configuration, dataset_path)
    
from sklearn.model_selection import train_test_split
if do_pre_train:
    pre_train_train_corpus, pre_train_test_corpus = train_test_split(pre_train_corpus, test_size=0.05, random_state=42)
if do_fine_tune or do_evaluate:
    train_benign_train_corpus, train_benign_test_corpus = train_test_split(train_benign_corpus, test_size=0.1, random_state=42)
    train_rt_train_corpus, train_rt_test_corpus = train_test_split(train_rt_corpus, test_size=0.1, random_state=42)

from bert_combined_dataset import *
if do_pre_train:
    pre_train_train_data = FlowPairDataset(configuration, pre_train_train_corpus, seq_length=seq_length, deduplicate=deduplicate, coalesce=coalesce, shuffle=True)
    pre_train_test_data = FlowPairDataset(configuration, pre_train_test_corpus, seq_length=seq_length, deduplicate=deduplicate, coalesce=coalesce, shuffle=True)
if do_fine_tune or do_evaluate:
    train_train_data = FlowPairDataset(configuration, train_rt_train_corpus, train_benign_train_corpus, seq_length=seq_length, deduplicate=True, coalesce=coalesce, shuffle=True, balanced=False)
    train_test_data = FlowPairDataset(configuration, train_rt_test_corpus, train_benign_test_corpus, seq_length=seq_length, deduplicate=True, coalesce=coalesce, shuffle=True, balanced=False)
    valid_valid_data = FlowPairDataset(configuration, valid_rt_corpus, valid_benign_corpus, seq_length=seq_length, deduplicate=True, coalesce=coalesce, shuffle=True, balanced=False)

from evaluate_bert import *
import os

finetuned_model_path = os.path.join(output_dir, fine_tune_run_name, "finetuned_model")
pretrained_model_path = os.path.join(output_dir, pre_train_run_name, "pretrained_model")

finetuned_model_path = "trained_models/fine_tune_2024_full_fi-100_ft-15000000_nb-30_ht-100000_di-uni_mx-1500_lg-no/finetuned_model_20_steps"

if use_finetuned_model:
    model_path = finetuned_model_path
else:
    model_path = pretrained_model_path

if do_evaluate:
    train_data, train_labels = get_embeddings(model_path=model_path, data=train_train_data, resample=False)
    test_data, test_labels = get_embeddings(model_path=model_path, data=train_test_data, resample=False)
    valid_data, valid_labels = get_embeddings(model_path=model_path, data=valid_valid_data, resample=False)
    
    train_data_rt = train_data[train_labels == 0]
    train_data_benign = train_data[train_labels == 1]
    test_data_rt = test_data[test_labels == 0]
    test_data_benign = test_data[test_labels == 1]
    valid_data_rt = valid_data[valid_labels == 0]
    valid_data_benign = valid_data[valid_labels == 1]

# load_path = "dataset_embeddings"
# train_data_rt = np.load(load_path + '/' + configuration + '_train_rt.npy')
# train_data_benign = np.load(load_path + '/' + configuration + '_train_benign.npy')
# test_data_rt = np.load(load_path + '/' + configuration + '_test_rt.npy')
# test_data_benign = np.load(load_path + '/' + configuration + '_test_benign.npy')
# valid_data_rt = np.load(load_path + '/' + configuration + '_valid_rt.npy')
# valid_data_benign = np.load(load_path + '/' + configuration + '_valid_benign.npy')

# pre_train_data_rt = np.load(load_path + '/pre_' + configuration + '_train_rt.npy')
# pre_train_data_benign = np.load(load_path + '/pre_' + configuration + '_train_benign.npy')
# pre_test_data_rt = np.load(load_path + '/pre_' + configuration + '_test_rt.npy')
# pre_test_data_benign = np.load(load_path + '/pre_' + configuration + '_test_benign.npy')
# pre_valid_data_rt = np.load(load_path + '/pre_' + configuration + '_valid_rt.npy')
# pre_valid_data_benign = np.load(load_path + '/pre_' + configuration + '_valid_benign.npy')
# print("Loaded embeddings from", load_path)









from evaluate_bert import *
pred_labels, pred_probas, true_labels = infer_label(fine_tuned_model_path=finetuned_model_path, data=valid_valid_data, batch_size=2048, use_labels=True)

from sklearn.metrics import classification_report

print("Classification Report Fine-Tuned Model:")
print(classification_report(true_labels, pred_labels, target_names=["RT", "Benign"]))













# # import random forest classifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score

# X = np.concatenate([pre_train_data_rt, pre_train_data_benign])
# y = np.concatenate([np.zeros(len(pre_train_data_rt)), np.ones(len(pre_train_data_benign))])
# X_test = np.concatenate([pre_test_data_rt, pre_test_data_benign])
# y_test = np.concatenate([np.zeros(len(pre_test_data_rt)), np.ones(len(pre_test_data_benign))])
# X_val = np.concatenate([pre_valid_data_rt, pre_valid_data_benign])
# y_val = np.concatenate([np.zeros(len(pre_valid_data_rt)), np.ones(len(pre_valid_data_benign))])

# clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0, n_jobs=-1)
# clf.fit(X, y)

# print("Pre-Trained Model (Random Forest)")

# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
# print("Accuracy:", accuracy_score(y_test, y_pred))

# y_pred = clf.predict(X_val)
# print(classification_report(y_val, y_pred))
# print(confusion_matrix(y_val, y_pred))
# print("Accuracy:", accuracy_score(y_val, y_pred))













# # import logistic regression
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score

# X = np.concatenate([pre_train_data_rt, pre_train_data_benign])
# y = np.concatenate([np.zeros(len(pre_train_data_rt)), np.ones(len(pre_train_data_benign))])
# X_test = np.concatenate([pre_test_data_rt, pre_test_data_benign])
# y_test = np.concatenate([np.zeros(len(pre_test_data_rt)), np.ones(len(pre_test_data_benign))])
# X_val = np.concatenate([pre_valid_data_rt, pre_valid_data_benign])
# y_val = np.concatenate([np.zeros(len(pre_valid_data_rt)), np.ones(len(pre_valid_data_benign))])

# clf = LogisticRegression(random_state=0, max_iter=1000, n_jobs=-1).fit(X, y)

# print("Pre-Trained Model (Logistic Regression)")

# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
# print("Accuracy:", accuracy_score(y_test, y_pred))

# y_pred = clf.predict(X_val)
# print(classification_report(y_val, y_pred))
# print(confusion_matrix(y_val, y_pred))
# print("Accuracy:", accuracy_score(y_val, y_pred))



















from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

X = np.concatenate([train_data_rt, train_data_benign])
y = np.concatenate([np.zeros(len(train_data_rt)), np.ones(len(train_data_benign))])
X_test = np.concatenate([test_data_rt, test_data_benign])
y_test = np.concatenate([np.zeros(len(test_data_rt)), np.ones(len(test_data_benign))])
X_val = np.concatenate([valid_data_rt, valid_data_benign])
y_val = np.concatenate([np.zeros(len(valid_data_rt)), np.ones(len(valid_data_benign))])

clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0, n_jobs=-1)
clf.fit(X, y)

print("Fine-Tuned Model (Random Forest)")

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
print("Accuracy:", accuracy_score(y_val, y_pred))























# import logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

X = np.concatenate([train_data_rt, train_data_benign])
y = np.concatenate([np.zeros(len(train_data_rt)), np.ones(len(train_data_benign))])
X_test = np.concatenate([test_data_rt, test_data_benign])
y_test = np.concatenate([np.zeros(len(test_data_rt)), np.ones(len(test_data_benign))])
X_val = np.concatenate([valid_data_rt, valid_data_benign])
y_val = np.concatenate([np.zeros(len(valid_data_rt)), np.ones(len(valid_data_benign))])

clf = LogisticRegression(random_state=0, max_iter=1000, n_jobs=-1).fit(X, y)

print("Fine-Tuned Model (Logistic Regression)")

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
print("Accuracy:", accuracy_score(y_val, y_pred))