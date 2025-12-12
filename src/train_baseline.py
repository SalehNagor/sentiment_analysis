import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.preprocessing import clean_text

logger = logging.getLogger(__name__)

def train_baseline(File_path):

    print("Loading and cleaning data...")
    column_names = ['a', 'b', 'sentiment', 'text']
    df = pd.read_csv(File_path,
    header=None,
    names=column_names)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df = df[df['sentiment'] != 'Irrelevant']

    df['text'] = df['text'].apply(clean_text)
    df['sentiment'] = df['sentiment'].replace({'Negative': 0, 'Neutral': 1, 'Positive': 2})

    X = df['text']  # Input features (Text)
    y = df['sentiment']  # Target variable (Sentiment)

    # Split the data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(solver='liblinear', multi_class='auto')
)

    print("Training the Baseline model...")
    model.fit(X_train, y_train)

    # Generate predictions on the test set
    preds = model.predict(X_test)

    # Print the results
    print("-" * 40)
    print(f"Baseline Accuracy: {accuracy_score(y_test, preds):.2f}")
    print("-" * 40)
    print("Detailed Classification Report:")
    logger.info(classification_report(y_test, preds))
    print("\n--- Classification Report ---")
    print(classification_report(y_test, preds))