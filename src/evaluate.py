import logging
import numpy as np
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)


def evaluate_model(trainer, test_dataset):
    """
    Evaluate the trained model on the test split and
    print + log accuracy and a full classification report.
    """
    logger.info("Running evaluation on test set...")
    print("Running evaluation on test set...")

    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    test_accuracy = eval_results.get("eval_accuracy", None)

    if test_accuracy is not None:
        logger.info("Test Accuracy: %.4f", test_accuracy)
        print(f"Test Accuracy: {test_accuracy:.4f}")
    else:
        logger.warning("eval_accuracy not found in eval_results: %s", eval_results)
        print("Warning: eval_accuracy not found in eval_results.")

    predictions, labels, _ = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions, axis=1)

    report = classification_report(
        labels,
        predicted_labels,
        target_names=["Negative", "Neutral", "Positive"],
    )

    logger.info("Classification report on test set:\n%s", report)
    print("\n--- Classification Report ---")
    print(report)

    return eval_results
