import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.preprocessing import load_and_split_data




def train_baseline(File_path):
    model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)), 
    LogisticRegression(solver='liblinear', multi_class='auto')
    )

    train_df, val_df, test_df = load_and_split_data(File_path)
    
    X_train = train_df['text']
    y_train = train_df['sentiment']
    X_test = test_df['text']
    y_test = test_df['sentiment']

    print("Training the Baseline model...")
    model.fit(X_train, y_train)

    # Generate predictions on the test set
    preds = model.predict(X_test)

    # Print the results
    print("-" * 40)
    print(f"Baseline Accuracy: {accuracy_score(y_test, preds):.2f}")
    print("-" * 40)
    print("Detailed Classification Report:")
    print(classification_report(y_test, preds))