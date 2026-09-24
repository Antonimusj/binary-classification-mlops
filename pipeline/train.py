import logging
import pickle
from pathlib import Path
import pandas as pd
from config.settings import (
    TARGET, MODEL_DIR, MODEL_LIST, TRAIN_SPLIT_FILE,
    COLUMNS_FILENAME, MODEL_NAME_PATTERN
)
from pipeline.training import train_logreg, train_rndforest, train_bayes

logger = logging.getLogger(__name__)

TRAINING_FUNCTIONS = {
    'logreg': train_logreg,
    'rndforest': train_rndforest,
    'bayes': train_bayes
}


def read_training_data():
    """Reads processed training dataset and splits it into feature matrix and target vector."""
    logger.info("\033[1;34m[TRAIN] Reading training dataset from %s...\033[0m", TRAIN_SPLIT_FILE)
    df = pd.read_csv(TRAIN_SPLIT_FILE)
    x_train = df.drop(columns=[TARGET])
    y_train = df[TARGET]
    return x_train, y_train


def train_model(x_train, y_train, model_name):
    """Executes model training for a specific algorithm and persists the trained pickle artifact."""
    logger.info("\033[1;34m[TRAIN] Training model: %s...\033[0m", model_name)
    train_func = TRAINING_FUNCTIONS[model_name]
    model = train_func(x_train, y_train)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    filename = MODEL_NAME_PATTERN.format(model_name=model_name)
    model_path = MODEL_DIR / filename
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("\033[1;32m[SUCCESS] Model %s trained and saved to %s\033[0m", model_name, model_path)
    return model_name


def run_training():
    """Executes training for all configured models and saves the feature column schema."""
    logger.info("\033[1;34m[TRAIN] Starting full training pipeline...\033[0m")
    x_train, y_train = read_training_data()
    for model_name in MODEL_LIST:
        train_model(x_train, y_train, model_name=model_name)
    columns_path = MODEL_DIR / COLUMNS_FILENAME
    logger.info("\033[1;34m[TRAIN] Saving feature column definitions to %s...\033[0m", columns_path)
    with open(columns_path, "wb") as f:
        pickle.dump(x_train.columns, f)
    logger.info("\033[1;32m[SUCCESS] Feature columns saved and training pipeline completed successfully.\033[0m")