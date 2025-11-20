# IMDB Sentiment Analysis Pipeline 

This project implements a modular Machine Learning pipeline for sentiment analysis on the IMDB movie reviews dataset. It utilizes **DistilBERT** for sequence classification, fine-tuned to achieve high accuracy in detecting positive and negative sentiments.

##  Project Features
- **Modular Architecture:** Code is organized into distinct modules for preprocessing, training, and evaluation.
- **End-to-End Pipeline:** A single entry point (`main.py`) manages the entire workflow from data ingestion to evaluation.
- **State-of-the-Art Model:** Uses Hugging Face's `DistilBertForSequenceClassification`.
- **Reproducibility:** Includes environment requirements and seed setting for consistent results.

##  Project Structure

```
project_root/
│
├── data/                   # Contains the dataset (IMDB_dataset.csv)
├── models/                 # Stores the fine-tuned model and checkpoints
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── preprocessing.py    # Data cleaning, splitting, and tokenization
│   ├── train.py            # Model initialization and training loop
│   └── evaluate.py         # Evaluation metrics and classification report
│
├── main.py                 # Main script to execute the pipeline
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
```

##  Installation

### 1. Clone the repository:

```bash
git clone [https://github.com/SalehNagor/sentiment_analysis.git](https://github.com/SalehNagor/sentiment_analysis.git)
cd sentiment_analysis
```

### 2. Create a virtual environment (Optional but recommended):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** Ensure you have accelerate installed for the Trainer API

##  Usage
To run the complete pipeline (Data Ingestion → Training → Evaluation), simply execute the `main.py` script:

```bash
python main.py
```

##  Results & Performance

The model was evaluated on a held-out test set representing **15% of the data** (7,437 reviews). On this test set, the model achieved an overall **Accuracy of ~88%**. To ensure robustness and verify balanced performance, we also utilized detailed metrics including **Precision**, **Recall**, and **F1-Score**:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative** | 0.88 | 0.88 | 0.88 | 3705 |
| **Positive** | 0.88 | 0.89 | 0.88 | 3732 |
| **Weighted Avg**| **0.88** | **0.88** | **0.88** | **7437** |

> **Note:** The balanced scores across both classes indicate that the model performs equally well on detecting positive and negative sentiments, with no significant bias.

##  Requirements
- **Python 3.8+**

- **Transformers**

- **PyTorch**

- **Scikit-learn**

- **Pandas**