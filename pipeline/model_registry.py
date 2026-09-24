import logging
import pickle
from config.settings import (
    MODEL_DIR, S3_BUCKET, REGISTRY_KEY,
    COLUMNS_REGISTRY_KEY, MODEL_NAME_PATTERN, COLUMNS_FILENAME, S3_REPLACE, SELECTION_METRIC
)
from pipeline.utils import s3_conection

logger = logging.getLogger(__name__)
hook = s3_conection()


def decide_model(results):
    """Selects the best performing model based on the configured selection metric."""
    best = max(results, key=lambda r: r[SELECTION_METRIC])
    logger.info(
        "\033[1;32m[SELECTION] Best model selected: %s | %s: %s\033[0m",
        best["model_name"],
        SELECTION_METRIC,
        best[SELECTION_METRIC]
    )
    return best


def model_registry(best):
    """Registers the best model artifact by uploading it to the production S3 registry path."""
    model_name = best['model_name']
    if model_name == "model.pkl":
        filename = model_name
    else:
        filename = MODEL_NAME_PATTERN.format(model_name=model_name)
    model_path = MODEL_DIR / filename
    hook.load_file(
        filename=str(model_path),
        key=REGISTRY_KEY,
        bucket_name=S3_BUCKET,
        replace=S3_REPLACE
    )
    logger.info("\033[1;32m[SUCCESS] Model successfully registered in S3.\033[0m")


def columns_registry():
    """Uploads the feature column schema (columns.pkl) to the production S3 registry."""
    file_path = MODEL_DIR / COLUMNS_FILENAME
    if not file_path.exists():
        logger.info(
            "\033[1;33m[SKIP] File columns.pkl not found locally. Skipping column registry update.\033[0m"
        )
        return
    with open(file_path, "rb") as f:
        columns = pickle.load(f)
    logger.info("\033[1;34m[UPLOAD] Uploading column registry to S3...\033[0m")
    hook.load_file(
        filename=str(file_path),
        key=COLUMNS_REGISTRY_KEY,
        bucket_name=S3_BUCKET,
        replace=S3_REPLACE
    )
    logger.info("\033[1;32m[SUCCESS] Column registry successfully updated in S3.\033[0m")