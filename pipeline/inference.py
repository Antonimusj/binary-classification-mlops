import logging
import pickle
from datetime import datetime
import pandas as pd
from config.settings import (
    INFERENCE_DIR, MODEL_DIR, PRED_FEATURES, COLUMNS_FILENAME,
    INFERENCE_INPUT_FILENAME, MODEL_FILENAME, INFERENCE_OUTPUT_PATTERN, INFERENCE_DATE_FORMAT
)
from pipeline.preprocess import preprocess
from pipeline.utils import upload_aws

logger = logging.getLogger(__name__)


def read_inference_data(input_path):
    """Reads inference raw CSV dataset and applies inference preprocessing transformations."""
    logger.info("\033[1;34m[INFERENCE] Reading and preprocessing inference data from %s...\033[0m", input_path)
    inference_df = pd.read_csv(input_path)
    return preprocess(inference_df, inference=True)


def run_inference():
    """Loads feature schema and trained model, generates predictions, saves outputs locally, and uploads to S3."""
    logger.info("\033[1;34m[INFERENCE] Starting inference process...\033[0m")

    columns_path = MODEL_DIR / COLUMNS_FILENAME
    logger.info("\033[1;34m[INFERENCE] Loading expected features columns from %s...\033[0m", columns_path)
    with open(columns_path, "rb") as f:
        columns = pickle.load(f)

    inference_date = datetime.now().astimezone().strftime(INFERENCE_DATE_FORMAT)
    output_filename = INFERENCE_OUTPUT_PATTERN.format(inference_date=inference_date)
    output_file = INFERENCE_DIR / output_filename

    input_file = INFERENCE_DIR / INFERENCE_INPUT_FILENAME
    inference_df = read_inference_data(input_file)

    model_path = MODEL_DIR / MODEL_FILENAME
    logger.info("\033[1;34m[INFERENCE] Loading model from %s...\033[0m", model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info("\033[1;34m[INFERENCE] Generating predictions...\033[0m")
    inference_df[PRED_FEATURES] = model.predict(inference_df)

    logger.info("\033[1;34m[INFERENCE] Saving inference results locally to %s...\033[0m", output_file)
    inference_df.to_csv(output_file, index=False)

    logger.info("\033[1;34m[INFERENCE] Uploading inference results to S3...\033[0m")
    upload_aws(output_file)
    logger.info("\033[1;32m[SUCCESS] Inference process completed successfully.\033[0m")