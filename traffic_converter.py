from utils import *

configuration = "fi-100_ft-15000000_nb-30_ht-100000_di-uni_mx-1500_lg-no"
dataset_path = "datasets"
model_path = "trained_models/fine_tune_all_years_ded_coa_med_seq-128_old_" + configuration + "/finetuned_model"

use_sentence_pairs = True

metadata_corpus = {}
ip_documents = {}

iteration = 0
while True:
    time.sleep(1)
    
    ips1, ips2, sentence_pairs = get_next_sentence_pairs("ls24_live_dataset/day_1", ip_documents)
    if len(sentence_pairs) == 0:
        continue
    
    sentence_dataset = SentencePairDataset(sentence_pairs, seq_length=128)
    embeddings = get_embeddings(model_path, sentence_dataset, batch_size=1024, resample=False, use_labels=False)
    host_embeddings, host_host_embeddings, ip1_list, ip2_list = update_embeddings(sentence_dataset, metadata_corpus, ips1, ips2, embeddings)
        
    similarity_2022_dataset = SimilarityDataset(2022, fine=True, label_path="ip_labels_2022.txt", only_rt=False)
    top_host_labels, top_host_simil = get_similarity_batch(host_embeddings, similarity_2022_dataset, is_host_host=False, batch_size=512, top_k=50)
    top_host_host_labels, top_host_host_simil = get_similarity_batch(host_host_embeddings, similarity_2022_dataset, is_host_host=True, batch_size=64, top_k=50)

    host_scores_rows = []
    for i, (ip, score, labels) in enumerate(zip(ip1_list, top_host_simil, top_host_labels)):
        for j in range(len(score)):
            host_scores_rows.append({"ip": ip, "score": score[j], "label": labels[j]})
    host_host_scores_rows = []
    for i, ((ip1, ip2), score, labels) in enumerate(zip(ip2_list, top_host_host_simil, top_host_host_labels)):
        for j in range(len(score)):
            host_host_scores_rows.append({"ip1": ip1, "ip2": ip2, "score": score[j], "label": labels[j]})
    
    for ip, scores, labels in zip(ip1_list, top_host_simil, top_host_labels):
        metadata_corpus[ip]["scores"] = scores
        metadata_corpus[ip]["labels"] = labels
    
    for (ip1, ip2), scores, labels in zip(ip2_list, top_host_host_simil, top_host_host_labels):
        metadata_corpus[ip1]["documents_meta"][ip2]["scores"] = scores
        metadata_corpus[ip1]["documents_meta"][ip2]["labels"] = labels
        
    joblib.dump(metadata_corpus, "live/metadata_corpus_live_large.pkl")
    print("Written metadata to file. Iteration: ", iteration, " Number of hosts: ", len(ip1_list), " Number of documents: ", len(ip2_list))
    iteration = iteration + 1