from transformers import BertConfig
from transformers import BertForPreTraining
from transformers import Trainer, TrainingArguments
import os
import json

def get_buckets(config):
    parts = config.split('_')
    num_buckets = 0
    for part in parts:
        if part.startswith('nb-'):
            num_buckets = int(part.split('-')[1])
    return num_buckets

def get_uni_directional(config):
    parts = config.split('_')
    uni_directional = False
    for part in parts:
        if part.startswith('di-'):
            uni_directional = part.split('-')[1] == 'uni'
    return uni_directional

def get_voc_size(config, offset=10):
    num_buckets = get_buckets(config)
    uni_directional = get_uni_directional(config)    
    if uni_directional:
        return num_buckets * 2 + offset
    return num_buckets + offset

def pre_train_bert(run_name, config, train_data, test_data, seq_length, embedding_length, num_hidden_layers, num_attention_heads, output_dir, num_epochs, per_device_train_batch_size, gradient_accumulation_steps=32):
    model_config = BertConfig(
        vocab_size=get_voc_size(config), 
        max_position_embeddings=seq_length,
        hidden_size=embedding_length,
        intermediate_size=4 * embedding_length,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
    )
    
    model = BertForPreTraining(config=model_config)
    model_path = os.path.join(output_dir, run_name)
    try:
        os.mkdir(model_path)
    except:
        pass

    training_args = TrainingArguments(
        output_dir=model_path,          # output directory to where save model checkpoint
        evaluation_strategy="steps",    # evaluate each `logging_steps` steps
        overwrite_output_dir=True,      
        num_train_epochs=num_epochs,            # number of training epochs, feel free to tweak
        per_device_train_batch_size=per_device_train_batch_size, # the training batch size, put it as high as your GPU memory fits
        gradient_accumulation_steps=gradient_accumulation_steps,  # accumulating the gradients before updating the weights
        per_device_eval_batch_size=per_device_train_batch_size,  # evaluation batch size
        logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
        save_steps=1000,
        greater_is_better=False,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        save_total_limit=2,
        run_name=run_name,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,   
    )
    trainer.train()
    
    save_path = os.path.join(model_path, 'pretrained_model')
    trainer.save_model(save_path)
    result = trainer.evaluate()

    hyperparameters = {
        'seq_length': seq_length,
        'embedding_length': embedding_length,
        'num_hidden_layers': num_hidden_layers,
        'num_attention_heads': num_attention_heads,
        'config': config,
        'run_name': run_name,
        'output_dir': model_path,
        'num_epochs': num_epochs,
        'per_device_train_batch_size': per_device_train_batch_size,
        'evaluation_result': result,
    }
    
    with open(os.path.join(model_path, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f, indent=4)
        
    return save_path