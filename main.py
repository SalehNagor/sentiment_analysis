import os
from src.preprocessing import load_and_split_data, tokenize_data
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    DATA_PATH = 'data/IMDB_dataset.csv' 
    MODEL_OUTPUT_DIR = './models/distilbert_finetuned'

    # Check data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found: {DATA_PATH}. Please put the csv in 'data' folder.")
        
    # Pipeline Steps
    train_df, val_df, test_df = load_and_split_data(DATA_PATH)
    tokenized_datasets, tokenizer = tokenize_data(train_df, val_df, test_df)

    model, trainer = train_model(tokenized_datasets, MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    evaluate_model(trainer, tokenized_datasets['test'])
    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    main()