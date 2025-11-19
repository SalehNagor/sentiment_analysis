import numpy as np
from sklearn.metrics import classification_report

def evaluate_model(trainer, test_dataset):
    print("Running evaluation on test set...")
    
    eval_results = trainer.evaluate(test_dataset)
    print(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}")

    predictions, labels, _ = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions, axis=1)

    print("\n--- Classification Report ---")
    print(classification_report(labels, predicted_labels, target_names=['Negative', 'Positive']))