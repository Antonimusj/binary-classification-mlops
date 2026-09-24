import logging
import pickle
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from config.settings import (
    TARGET, MODEL_LIST, MODEL_DIR, MODEL_NAME_PATTERN,
    DATE_FORMAT, ENCODED_FEATURES
)
from pipeline.preprocess import preprocess

logger = logging.getLogger(__name__)


def read_test_data(input_path):
    """Reads test or validation dataset, applies preprocessing if necessary, and splits features and target."""
    real_df = pd.read_csv(input_path)
    real_df.columns = real_df.columns.str.strip()
    if ENCODED_FEATURES[0] in real_df.columns:
        real_df = preprocess(real_df, inference=False)
    elif TARGET not in real_df.columns:
        raise ValueError(
            f"\033[1;31m[ERROR] Invalid dataset. Missing raw column '{ENCODED_FEATURES[0]}' and Target '{TARGET}'. Columns: {list(real_df.columns)}\033[0m"
        )
    x_test = real_df.drop(columns=[TARGET])
    y_test = real_df[TARGET]
    return x_test, y_test


def evaluate_model(x_test, y_test, model_name, training_date):
    """Loads a trained model artifact and evaluates its predictions against ground truth labels."""
    logger.info("\033[1;34m[EVAL] Evaluating model: %s\033[0m", model_name)
    if model_name == "model.pkl":
        model_path = MODEL_DIR / model_name
    else:
        filename = MODEL_NAME_PATTERN.format(model_name=model_name)
        model_path = MODEL_DIR / filename
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    predictions = model.predict(x_test)
    acc = accuracy_score(y_test, predictions)
    roc = roc_auc_score(y_test, predictions)
    return {
        "model_name": model_name,
        "training_date": training_date,
        "accuracy": acc,
        "roc_auc": roc
    }


def run_evaluation(input_path):
    """Runs evaluation for all candidate models or falls back to the production model from S3."""
    results = []
    x_test, y_test = read_test_data(input_path)
    training_date = datetime.now().astimezone().strftime(DATE_FORMAT)
    for model_name in MODEL_LIST:
        try:
            result = evaluate_model(x_test, y_test, model_name, training_date)
            results.append(result)
        except FileNotFoundError:
            logger.warning(
                f"\033[1;33m[WARN] Candidate model '{model_name}' not found locally. Evaluating model from S3.\033[0m"
            )
            model_name = "model.pkl"
            result = evaluate_model(x_test, y_test, model_name, training_date)
            results.append(result)
            return results
    if not results:
        logger.error("\033[1;31m[ERROR] No models were found locally for evaluation!\033[0m")
    return results


def process_performance_check(results, accuracy_threshold):
    """Processes evaluation metrics and determines the appropriate pipeline execution path."""
    if not results:
        logger.error("\033[1;31m[ERROR] No results were found for evaluation!\033[0m")
        return "low_performance_alert"
    for res in results:
        m_name = res.get("model_name", "Unknown")
        acc = res.get("accuracy", 0.0)
        logger.info(
            f"\033[1;34m[METRICS] -> Model: {m_name} | Calculated Accuracy: {acc:.4f} (Expected Minimum: {accuracy_threshold:.4f})\033[0m"
        )
    max_accuracy = max(res["accuracy"] for res in results)
    if max_accuracy < accuracy_threshold:
        logger.warning(
            f"\033[1;33m[WARN] Performance {max_accuracy:.4f} BELOW threshold {accuracy_threshold}. Triggering RETRAINING!\033[0m"
        )
        return "low_performance_alert"
    logger.info(
        f"\033[1;32m[SUCCESS] Performance {max_accuracy:.4f} STABLE and within expectations. Skipping retraining.\033[0m"
    )
    return "stable_performance"