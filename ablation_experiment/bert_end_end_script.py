configuration = "fi-100_ft-15000000_nb-30_ht-100000_di-uni_mx-1500_lg-no"
dataset_path = "datasets"
pre_train_years = [2017, 2018, 2019, 2020]
train_year = 2022
validation_year = 2021
output_dir = "trained_models"
do_fine_tune = False


#### Pre-Processing:
seq_length = 256
deduplicate = True
coalesce = True


#### Pre-Training:
pre_train_run_name = "pre_train_all_years_ded_coa_large_seq-256" + configuration
# for largest configuration: use batch size of 48 on A100, 16 on others
# medium configuration: 128 on A100, 32 on others
per_device_batch_size = 128 # A100 up to 256, V100 and titanv up to 64
embedding_length = 512
num_attention_heads = 8
num_hidden_layers = 8
num_epochs = 2
print(pre_train_run_name)


################################# Loading Data ########################################################
from data_loader import *
pre_train_corpus = load_pretrain_data(pre_train_years, configuration, dataset_path)
if do_fine_tune:
    train_benign_corpus, train_rt_corpus = load_train_data(train_year, configuration, dataset_path)
    valid_benign_corpus, valid_rt_corpus = load_train_data(validation_year, configuration, dataset_path)
    
    
    
################################# Creating Splits ########################################################
from sklearn.model_selection import train_test_split
pre_train_train_corpus, pre_train_test_corpus = train_test_split(pre_train_corpus, test_size=0.05, random_state=42)
if do_fine_tune:
    train_benign_train_corpus, train_benign_test_corpus = train_test_split(train_benign_corpus, test_size=0.2, random_state=42)
    train_rt_train_corpus, train_rt_test_corpus = train_test_split(train_rt_corpus, test_size=0.2, random_state=42)
    

################################# Create Datasets ########################################################
from bert_combined_dataset import *
pre_train_train_data = FlowPairDataset(configuration, pre_train_train_corpus, seq_length=seq_length, deduplicate=deduplicate, coalesce=coalesce, shuffle=True)
pre_train_test_data = FlowPairDataset(configuration, pre_train_test_corpus, seq_length=seq_length, deduplicate=deduplicate, coalesce=coalesce, shuffle=True)
if do_fine_tune:
    train_train_data = FlowPairDataset(configuration, train_rt_train_corpus, train_benign_train_corpus, seq_length=seq_length, deduplicate=deduplicate, coalesce=coalesce, shuffle=True)
    train_test_data = FlowPairDataset(configuration, train_rt_test_corpus, train_benign_test_corpus, seq_length=seq_length, deduplicate=deduplicate, coalesce=coalesce, shuffle=True)
    valid_data = FlowPairDataset(configuration, valid_rt_corpus, valid_benign_corpus, seq_length=seq_length, deduplicate=deduplicate, coalesce=coalesce, shuffle=True)



################################# Pre-Train BERT ########################################################
from pre_train_bert import *
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