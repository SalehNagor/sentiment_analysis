import os
import logging
from src.preprocessing import load_and_split_data, tokenize_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.logging_utils import setup_logging
from src.train_baseline import train_baseline

def main():
    # Configure logging once at startup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Pipeline started.")

    # اسم ملف الداتا داخل مجلد data (مسار نسبي)
    DATA_PATH = "data/twitter_training.csv"
    MODEL_OUTPUT_DIR = "./models/distilbert_finetuned"

    # Check data
    if not os.path.exists(DATA_PATH):
        logger.error("File not found: %s", DATA_PATH)
        raise FileNotFoundError(
            f"File not found: {DATA_PATH}. Please put the csv in 'data' folder."
        )

    # 1) Load and split data
    logger.info("Loading and splitting data from %s", DATA_PATH)
    train_df, val_df, test_df = load_and_split_data(DATA_PATH)

    # 2) Tokenize
    logger.info("Tokenizing datasets.")
    tokenized_datasets, tokenizer = tokenize_data(train_df, val_df, test_df)

    # 3) Train model
    logger.info("Starting model training.")
    model, trainer = train_model(tokenized_datasets, MODEL_OUTPUT_DIR)
    logger.info("Training completed. Saving tokenizer to %s", MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    # 4) Evaluate on test set
    logger.info("Starting evaluation on test split.")
    evaluate_model(trainer, tokenized_datasets["test"])

    logger.info("Pipeline execution completed successfully.")
    print("\nPipeline execution completed successfully!")
    
  

if __name__ == "__main__":
    main()