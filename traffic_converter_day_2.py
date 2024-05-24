from utils import *
from evaluate_bert import *
from datetime import datetime

configuration = "fi-100_ft-15000000_nb-30_ht-100000_di-uni_mx-1500_lg-no"
dataset_path = "datasets"
model_path = "trained_models/fine_tune_2024_full_fi-100_ft-15000000_nb-30_ht-100000_di-uni_mx-1500_lg-no/finetuned_model_20_steps"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to('cuda')
model.eval()

use_sentence_pairs = True

ip_documents = {}
iteration = 0

def update_metadata(labels, ips1, ips2, probas):
    columns=["timestamp", "ip1", "ip2", "label", "prob"]
    timestamps = [datetime.now()] * len(labels)
    metadata_file_path = "live/predictions_777.csv"
    pd.DataFrame(list(zip(timestamps, ips1, ips2, labels, probas)), columns=columns).to_csv(metadata_file_path, mode='a', header=False, index=False)

while True:
    time.sleep(1)
    
    ips1, ips2, sentence_pairs = get_next_sentence_pairs("ls24_live_dataset/last_phase", ip_documents)
    if len(sentence_pairs) == 0:
        continue
    
    sentence_dataset = SentencePairDataset(sentence_pairs, seq_length=128)
    labels, probas = infer_label(model_path, sentence_dataset, 2048, softmax_threshold=0.5)
    
    update_metadata(labels, ips1, ips2, probas)
    print("Written metadata to file. Iteration: ", iteration)

    iteration = iteration + 1