import numpy as np
import os
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=1)
    accuracy = (preds == labels).mean()
    return {'accuracy': accuracy}

def train_model(tokenized_datasets, output_dir):
    print("Initializing model and training...")
    os.environ["WANDB_DISABLED"] = "true"

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

    training_args = TrainingArguments(
        output_dir='./models/checkpoints',
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