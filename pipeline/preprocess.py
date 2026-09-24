import logging
import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config.settings import (
    TEST_SIZE, RANDOM_STATE, TARGET, FEATURES, ENCODED_FEATURES,
    MODEL_DIR, COLUMNS_FILENAME, FILL_VALUE_INFERENCE
)

logger = logging.getLogger(__name__)


def preprocess(df, inference=False):
    """Encodes categorical features, target variable, and aligns feature schema for inference."""
    logger.info("\033[1;34m[PREPROCESS] Starting preprocessing (inference=%s)...\033[0m", inference)
    cols = FEATURES if inference else FEATURES + [TARGET]
    df = df[cols].copy()
    if not inference and TARGET in df.columns:
        le = LabelEncoder()
        df[TARGET] = le.fit_transform(df[TARGET])
    df = pd.get_dummies(df, columns=ENCODED_FEATURES)
    df = df.astype({col: int for col in df.columns if df[col].dtype == bool})
    if inference:
        columns_path = MODEL_DIR / COLUMNS_FILENAME
        logger.info("\033[1;34m[PREPROCESS] Aligning inference columns with train columns from %s...\033[0m", columns_path)
        with open(columns_path, "rb") as f:
            train_columns = pickle.load(f)
        df = df.reindex(columns=train_columns, fill_value=FILL_VALUE_INFERENCE)
    logger.info("\033[1;32m[SUCCESS] Preprocessing completed successfully.\033[0m")
    return df


def preprocess_data(input_path, output_path):
    """Reads raw CSV dataset, applies preprocessing pipeline, and saves processed output."""
    logger.info("\033[1;34m[PREPROCESS] Reading data from %s...\033[0m", input_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(input_path)
    preprocess(train_df).to_csv(output_path, index=False)
    logger.info("\033[1;32m[SUCCESS] Preprocessed data saved to %s\033[0m", output_path)


def split_data(input_file, train_output_file, test_output_file):
    """Splits processed dataset into train and test CSV files."""
    logger.info("\033[1;34m[SPLIT] Splitting dataset into train/test sets...\033[0m")
    Path(train_output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(test_output_file).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_file)
    x_train, x_test, y_train, y_test = make_test_train(df)
    train_df = pd.DataFrame(x_train)
    train_df[TARGET] = y_train
    test_df = pd.DataFrame(x_test)
    test_df[TARGET] = y_test
    test_df.to_csv(test_output_file, index=False)
    train_df.to_csv(train_output_file, index=False)
    logger.info("\033[1;32m[SUCCESS] Train data saved to %s and test data saved to %s\033[0m", train_output_file, test_output_file)


def make_test_train(df):
    """Splits dataframe features and target into train and test sets using train_test_split."""
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)