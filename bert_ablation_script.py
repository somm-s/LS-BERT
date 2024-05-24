configuration = "fi-100_ft-15000000_nb-30_ht-100000_di-bi_mx-1500_lg-no"
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
do_pre_train = True
pre_train_run_name = "pre_train_all_years_ded_coa_med_mini_seq-128_abl_" + configuration
# for largest configuration: use batch size of 48 on A100, 16 on others
# medium configuration: 128 on others
per_device_batch_size = 128
embedding_length = 64
num_attention_heads = 8
num_hidden_layers = 8
num_epochs = 2

#### Fine-Tuning:
do_fine_tune = True
fine_tune_run_name = "fine_tune_all_years_ded_coa_med_mini_seq-128_abl_" + configuration
num_fine_tune_epochs = 1
fine_tune_per_device_batch_size = 128
fine_tune_per_device_eval_batch_size = 256


########################################################################################################

from data_loader import *
if do_pre_train:
    pre_train_corpus = load_pretrain_data(pre_train_years, configuration, dataset_path)
if do_fine_tune:
    train_benign_corpus, train_rt_corpus = load_train_data(train_year, configuration, dataset_path)
    valid_benign_corpus, valid_rt_corpus = load_train_data(validation_year, configuration, dataset_path)
    
from sklearn.model_selection import train_test_split
if do_pre_train:
    pre_train_train_corpus, pre_train_test_corpus = train_test_split(pre_train_corpus, test_size=0.05, random_state=42)
if do_fine_tune:
    train_benign_train_corpus, train_benign_test_corpus = train_test_split(train_benign_corpus, test_size=0.1, random_state=42)
    train_rt_train_corpus, train_rt_test_corpus = train_test_split(train_rt_corpus, test_size=0.1, random_state=42)
    
from bert_combined_dataset import *
if do_pre_train:
    pre_train_train_data = FlowPairDataset(configuration, pre_train_train_corpus, seq_length=seq_length, deduplicate=deduplicate, coalesce=coalesce, shuffle=True)
    pre_train_test_data = FlowPairDataset(configuration, pre_train_test_corpus, seq_length=seq_length, deduplicate=deduplicate, coalesce=coalesce, shuffle=True)
if do_fine_tune:
    train_train_data = FlowPairDataset(configuration, train_rt_train_corpus, train_benign_train_corpus, seq_length=seq_length, deduplicate=True, coalesce=coalesce, shuffle=True, balanced=True)
    train_test_data = FlowPairDataset(configuration, train_rt_test_corpus, train_benign_test_corpus, seq_length=seq_length, deduplicate=True, coalesce=coalesce, shuffle=True, balanced=True)
    valid_data = FlowPairDataset(configuration, valid_rt_corpus, valid_benign_corpus, seq_length=seq_length, deduplicate=True, coalesce=coalesce, shuffle=True, balanced=True)

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
        eval_data=valid_data,
        pretrain_run_name=pre_train_run_name,
        per_device_train_batch_size=fine_tune_per_device_batch_size,
        num_epochs=num_fine_tune_epochs,
        per_device_eval_batch_size=fine_tune_per_device_eval_batch_size,
        logging_steps=1,
        save_steps=1,
    )