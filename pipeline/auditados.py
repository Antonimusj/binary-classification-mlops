import os
from pathlib import Path
import boto3
import numpy as np
import pandas as pd

# ==============================================================================
# 1. S3 & ENVIRONMENT CONFIGURATIONS
# ==============================================================================
# S3 Bucket and prefix definitions
S3_BUCKET = "antonimus-bucket"
S3_OUTPUT_PREFIX = "projeto-ml/output/"

# S3 keys expected by the Airflow DAG
S3_TRAIN_KEY = "projeto-ml/train/train.csv"
S3_REAL_DATA_KEY = "projeto-ml/real_data/Employee.csv"

# Local temporary directory for notebook processing
TMP_DIR = Path("./tmp_auditoria")
TMP_DIR.mkdir(exist_ok=True)

s3_client = boto3.client("s3")

# ==============================================================================
# 2. CONSOLIDATE INFERENCE OUTPUTS FROM S3
# ==============================================================================
# List and download all inference result CSV files from S3
paginator = s3_client.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_OUTPUT_PREFIX)

csv_files = []
for page in pages:
    if "Contents" in page:
        for obj in page["Contents"]:
            if obj["Key"].endswith(".csv"):
                csv_files.append(obj["Key"])

dataframes = []
for key in csv_files:
    file_name = key.split("/")[-1]
    local_path = TMP_DIR / file_name
    s3_client.download_file(S3_BUCKET, key, str(local_path))
    dataframes.append(pd.read_csv(local_path))

consolidated_df = pd.concat(dataframes, ignore_index=True)

# Drop predicted target column to prepare for ground truth labeling
if "LeaveOrNot_predicted" in consolidated_df.columns:
    consolidated_df = consolidated_df.drop(columns=["LeaveOrNot_predicted"])

# Labeling / Ground Truth Assignment (Simulated placeholder for actual audit logic)
np.random.seed(42)
consolidated_df["LeaveOrNot"] = np.random.choice([0, 1], size=len(consolidated_df), p=[0.6, 0.4])
print(f"Total newly audited records: {len(consolidated_df)} rows.")

# ==============================================================================
# 3. S3 DATASETS UPDATE & UPLOAD FUNCTION
# ==============================================================================
def update_and_upload_dataset(s3_key, new_df, local_filename):
    """Downloads existing S3 dataset, appends newly audited data, and updates S3."""
    local_path = TMP_DIR / local_filename

    # 1. Try downloading existing dataset from S3 to append new records
    try:
        s3_client.download_file(S3_BUCKET, s3_key, str(local_path))
        existing_df = pd.read_csv(local_path)
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"Data append successful for {s3_key}.")
    except Exception:
        print(f"Existing dataset not found at {s3_key}. Creating new file.")
        final_df = new_df

    # 2. Save updated dataset locally
    final_df.to_csv(local_path, index=False)

    # 3. Upload updated dataset to replace previous S3 file
    s3_client.upload_file(str(local_path), S3_BUCKET, s3_key)
    print(f"Dataset successfully updated in S3: {s3_key} | Total rows: {len(final_df)}")


# Execute dataset updates in S3 for both validation and training sets
update_and_upload_dataset(S3_REAL_DATA_KEY, consolidated_df, "Employee.csv")
update_and_upload_dataset(S3_TRAIN_KEY, consolidated_df, "train.csv")