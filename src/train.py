import numpy as np
import os
import torch
import torch.nn.utils.prune as prune
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=1)
    accuracy = (preds == labels).mean()
    return {'accuracy': accuracy}

def train_model_before_pruning(tokenized_datasets, output_dir):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==================================================")
    print(f"Status: Training will run on: {device.upper()}")
    if device == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"==================================================")

    print("Initializing model and training...")
    os.environ["WANDB_DISABLED"] = "true"

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

    training_args = TrainingArguments(
        output_dir='./models/checkpoints_before_pruning',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=['none']
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    return model, trainer


#==================================================================================================
#==================================================================================================



def prune_and_finetune(model, trainer, tokenized_datasets, output_dir, amount=0.3):
   
    print(f"\n{'='*50}")
    print(f"Status: Applying Pruning (Amount: {amount*100}%)")
    print(f"{'='*50}")

    #  Apply Pruning ...
    # We target 'Linear' layers because they contain most of the parameters in Transformers.
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    # Global Unstructured Pruning:
    # Removes the lowest x% of weights across ALL selected layers collectively.
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    # Check and print the actual sparsity (percentage of zeros)
    print("Checking sparsity...")
    total_zeros = 0
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            total_zeros += torch.sum(module.weight == 0)
            total_params += module.weight.nelement()
    
    print(f"Sparsity in Linear layers: {float(total_zeros)/float(total_params)*100:.2f}%")

    #  Fine-tuning (Retraining) ...
    print("\nStarting Fine-tuning (Retraining) to recover accuracy...")
    
    # We update the trainer settings for a short retraining session.
    # We lower the learning rate to carefully adjust the remaining weights.
    trainer.args.num_train_epochs = 1 
    trainer.args.learning_rate = 2e-5 
    trainer.args.output_dir = './models/checkpoints_pruning_finetune'
    
    # Start the retraining process
    trainer.train()

    # Make Pruning Permanent ...
    # Currently, pruning is just a "mask" over the weights. 
    # We need to apply the mask to the weights to make the zeros permanent.
    print("Removing pruning masks (making zeros permanent)...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')

    # Save the final Pruned Model
    print(f"Saving Pruned model to {output_dir}...")
    model.save_pretrained(output_dir)
    
    return model, trainer