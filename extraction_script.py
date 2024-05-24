import os

ip_documents = {}

def get_next_sentence_pairs(document_path):    
    labelled_sentence_pairs = []
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
                    labelled_sentence_pairs.append((last_sentence, sentence, ip1, ip2, start_time, end_time))
                else:
                    ip_documents[ip1][ip2] = sentence
                    labelled_sentence_pairs.append((sentence, "3 \n", ip1, ip2, start_time, end_time))
                    
    return labelled_sentence_pairs



######### Repeat from here: #########
from bert_inference_dataset import *

sentence_pairs = get_next_sentence_pairs("ls24_live_dataset/exploration_revert")
data = BERTInferenceDataset(sentence_pairs, seq_length=128)

from evaluate_bert import *
model_path = "trained_models/fine_tune_all_years_ded_coa_med_seq-128fi-100_ft-15000000_nb-30_ht-100000_di-uni_mx-1500_lg-no/finetuned_model"
embeddings = get_embeddings(model_path, data, batch_size=1024, resample=False, use_labels=False)

import joblib
svm = joblib.load("svm_model.pkl")

batch_size = 5000
bt_embeddings = []
rt_embeddings = []
labels = []
for i in range(0, embeddings.shape[0], batch_size):
    print("Processing embedding ", i, " of ", embeddings.shape[0])
    predictions = svm.predict_proba(embeddings[i:i+batch_size])[:,1]
    y_pred_batch = (predictions >= 0.01).astype(int)
    rt_embeddings_batch = embeddings[i:i+batch_size][y_pred_batch == 0]
    bt_embeddings_batch = embeddings[i:i+batch_size][y_pred_batch == 1]
    
    labels.extend(y_pred_batch)
    rt_embeddings.extend(rt_embeddings_batch)
    bt_embeddings.extend(bt_embeddings_batch)

columns = ["external_host", "internal_host", "start_time", "end_time", "label"]
file_name = "ls24_live_dataset/predictions.csv"
all_data_file_name = "ls24_live_dataset/all_data.csv"

with open(file_name, 'a') as f:
    with open(all_data_file_name, 'a') as f2:
        for label, metadata in zip(labels, data.metadata):
            if label == 0:
                ip1, ip2, start_time, end_time = metadata
                f2.write(ip1 + "," + ip2 + "," + start_time + "," + end_time + "\n")
                f.write(ip1 + "," + ip2 + "," + start_time + "," + end_time + ",0" + "\n")
            else:
                ip1, ip2, start_time, end_time = metadata
                f2.write(ip1 + "," + ip2 + "," + start_time + "," + end_time + ",1" + "\n")