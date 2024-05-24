# configuration = "fi-100_ft-15000000_nb-30_ht-100000_di-uni_mx-1500_lg-no"
# configuration = "fi-100_ft-15000000_nb-30_ht-100000_di-uni_mx-11_lg-yes"
# configuration = "fi-100_ft-15000000_nb-30_ht-100000_di-bi_mx-1500_lg-no"
# configuration = "fi-100_ft-15000000_nb-30_ht-0_di-uni_mx-1500_lg-no"
configuration = "fi-100_ft-15000000_nb-20_ht-100000_di-uni_mx-1500_lg-no"

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
num_fine_tune_epochs = 1
fine_tune_per_device_batch_size = 128
fine_tune_per_device_eval_batch_size = 256


#### Evaluation:
do_evaluate = True
use_finetuned_model = False

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

from pre_train_bert import *
if do_pre_train:
    pretrained_model_path = pre_train_bert(
        run_name=pre_train_run_name,
        config=configuration,
        train_data=pre_train_train_data,
        test_data=pre_train_test_data,
        seq_length=seq_length,
        num_epochs=num_epochs,
        num_hidden_layers= num_hidden_layers,
        num_attention_heads=num_attention_heads,
        embedding_length=embedding_length,
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
    )
    
from fine_tune_bert import *
if do_fine_tune:
    fine_tune_bert(
        run_name=fine_tune_run_name,
        config=configuration,
        output_dir=output_dir,
        train_data=train_train_data,
        eval_data=valid_valid_data,
        pretrain_run_name=pre_train_run_name,
        per_device_train_batch_size=fine_tune_per_device_batch_size,
        num_epochs=num_fine_tune_epochs,
        per_device_eval_batch_size=fine_tune_per_device_eval_batch_size,
        logging_steps=1,
        save_steps=1,
    )
    
from evaluate_bert import *
finetuned_model_path = os.path.join(output_dir, fine_tune_run_name, "finetuned_model")
pretrained_model_path = os.path.join(output_dir, pre_train_run_name, "pretrained_model")
if use_finetuned_model:
    model_path = finetuned_model_path
else:
    model_path = pretrained_model_path
    
train_data, train_labels = get_embeddings(model_path=model_path, data=train_train_data, resample=False)
test_data, test_labels = get_embeddings(model_path=model_path, data=train_test_data, resample=False)
valid_data, valid_labels = get_embeddings(model_path=model_path, data=valid_valid_data, resample=False)

train_data_rt = train_data[train_labels == 0]
train_data_benign = train_data[train_labels == 1]
test_data_rt = test_data[test_labels == 0]
test_data_benign = test_data[test_labels == 1]
valid_data_rt = valid_data[valid_labels == 0]
valid_data_benign = valid_data[valid_labels == 1]

save_path = "dataset_embeddings"
np.save(save_path + '/pre_' + configuration + '_train_rt.npy', train_data_rt)
np.save(save_path + '/pre_' + configuration + '_train_benign.npy', train_data_benign)
np.save(save_path + '/pre_' + configuration + '_test_rt.npy', test_data_rt)
np.save(save_path + '/pre_' + configuration + '_test_benign.npy', test_data_benign)
np.save(save_path + '/pre_' + configuration + '_valid_rt.npy', valid_data_rt)
np.save(save_path + '/pre_' + configuration + '_valid_benign.npy', valid_data_benign)
print("Saved embeddings to", save_path)
