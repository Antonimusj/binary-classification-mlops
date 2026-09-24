import logging
import pandas as pd
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from config.settings import S3_BUCKET, OUTPUT_KEY, AWS_CONN_ID, AUDITED_KEY
from pipeline.preprocess import preprocess, preprocess_data

logger = logging.getLogger(__name__)


def s3_conection():
    """Instantiates and returns an S3Hook using the configured AWS connection ID."""
    return S3Hook(aws_conn_id=AWS_CONN_ID)


def upload_aws(input_file):
    """Uploads a local file to the specified S3 bucket and output prefix."""
    hook = s3_conection()
    s3_key = f"{OUTPUT_KEY}/{input_file.name}"
    logger.info(
        "\033[1;34m[UPLOAD] Uploading file %s to S3 bucket %s (key: %s)...\033[0m",
        input_file.name,
        S3_BUCKET,
        s3_key,
    )
    hook.load_file(
        filename=str(input_file),
        key=s3_key,
        bucket_name=S3_BUCKET,
        replace=True,
    )
    logger.info(
        "\033[1;32m[SUCCESS] File %s successfully uploaded to S3.\033[0m",
        input_file.name,
    )


def consolidate_train_data(raw_train_path, audited_path, output_path):
    """Preprocesses raw training data, concatenates it with audited production data, and saves the result."""
    logger.info(
        "\033[1;34m[CONSOLIDATE] Starting consolidated data preparation for retraining...\033[0m"
    )
    preprocess_data(raw_train_path, output_path)
    df_train_encoded = pd.read_csv(output_path)
    df_audited_encoded = pd.read_csv(audited_path)
    df_final = pd.concat(
        [df_train_encoded, df_audited_encoded], ignore_index=True
    )
    df_final.to_csv(output_path, index=False)
    logger.info(
        f"\033[1;32m[SUCCESS] Successfully consolidated encoded training data to: {output_path}\033[0m"
    )
    return output_path


def sensor():
    """Checks for the presence of the audited dataset in S3 to route execution to training or monitoring."""
    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    logger.info(
        f"\033[1;34m[SENSOR] Checking presence of file in S3: {AUDITED_KEY}\033[0m"
    )
    if hook.check_for_key(key=AUDITED_KEY, bucket_name=S3_BUCKET):
        logger.info(
            "\033[1;32m[SENSOR] Audit file found in S3! Proceeding to audited evaluation.\033[0m"
        )
        return "evaluate_audited_pipeline"
    else:
        logger.info(
            "\033[1;33m[SENSOR] Audit file not found in S3. Proceeding to initial training.\033[0m"
        )
        return "train_pipeline"