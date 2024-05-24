from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import os
from transformers import get_linear_schedule_with_warmup
import json


def fine_tune_bert(run_name, config, train_data, eval_data, output_dir, pretrain_run_name, per_device_train_batch_size=96, num_epochs=1, per_device_eval_batch_size=256, logging_steps=500, save_steps=500):
    
    pretrained_model_path = os.path.join(output_dir, pretrain_run_name, "pretrained_model")    
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=2)

    finetune_model_path = os.path.join(output_dir, run_name)
    try:
        os.mkdir(finetune_model_path)
    except:
        pass

    training_args = TrainingArguments(
        output_dir=finetune_model_path,          # output directory to where save model checkpoint
        evaluation_strategy="steps",    # evaluate each `logging_steps` steps
        overwrite_output_dir=True,      
        num_train_epochs=num_epochs,            # number of training epochs, feel free to tweak
        per_device_train_batch_size=per_device_train_batch_size, # the training batch size, put it as high as your GPU memory fits
        gradient_accumulation_steps=32,  # accumulating the gradients before updating the weights
        per_device_eval_batch_size=per_device_eval_batch_size,  # evaluation batch size
        logging_steps=logging_steps,             # evaluate, log and save model checkpoints every 1000 step
        save_steps=save_steps,
        greater_is_better=False,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        save_total_limit=3,
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )
    trainer.train()
    
    save_path = os.path.join(finetune_model_path, 'finetuned_model')
    trainer.save_model(save_path)
    print(f"Model saved to {save_path}")
    result = trainer.evaluate()

    hyperparameters = {
        'config': config,
        'run_name': run_name,
        'output_dir': finetune_model_path,
        'num_epochs': num_epochs,
        'per_device_train_batch_size': per_device_train_batch_size,
        'per_device_eval_batch_size': per_device_eval_batch_size,
        'logging_steps': logging_steps,
        'save_steps': save_steps,
        'evaluation_result': result,
        'pretrain_run_name': pretrain_run_name,
    }
    
    with open(os.path.join(finetune_model_path, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f)