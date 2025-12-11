import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from datasets import Dataset, DatasetDict

def clean_text(text):
    """Clean HTML and keep only alphabets."""
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def load_and_split_data(csv_path):
    print("Loading and cleaning data...")
    column_names = ['a', 'b', 'sentiment', 'text']
    df = pd.read_csv(csv_path, 
    header=None,  
    names=column_names)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    df['text'] = df['text'].apply(clean_text)
    df['sentiment'] = df['sentiment'].replace({'Negative': 0, 'Neutral': 1, 'Positive': 2})

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['sentiment'])
    test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['sentiment'])
    
    return train_df, val_df, test_df

def tokenize_data(train_df, val_df, test_df):
    print("Tokenizing data...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=128
        )
        tokenized_inputs['labels'] = examples['sentiment']
        return tokenized_inputs

    tokenized_datasets = DatasetDict({
        'train': train_dataset.map(tokenize_function, batched=True),
        'validation': val_dataset.map(tokenize_function, batched=True),
        'test': test_dataset.map(tokenize_function, batched=True)
    })
    
    return tokenized_datasets, tokenizer