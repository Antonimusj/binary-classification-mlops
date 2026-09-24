import os
import shutil
from airflow.operators.python import PythonOperator
from airflow import DAG
from config.settings import (
    DATA_DIR,
    INFERENCE_DAG_ID,
    INFERENCE_DAG_START_DATE,
    INFERENCE_DAG_SCHEDULE,
    INFERENCE_DAG_CATCHUP,
)
from pipeline.data_loader import inference_load, load_model, load_columns
from pipeline.inference import run_inference


def cleanup():
    """Removes existing local data directory and recreates an empty one."""
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
        os.makedirs(DATA_DIR, exist_ok=True)

with DAG(
    dag_id=INFERENCE_DAG_ID,
    start_date=INFERENCE_DAG_START_DATE,
    schedule=INFERENCE_DAG_SCHEDULE,
    catchup=INFERENCE_DAG_CATCHUP,
) as dag:

    task0 = PythonOperator(
        task_id="cleanup",
        python_callable=cleanup,
    )

    task1 = PythonOperator(
        task_id="load_columns",
        python_callable=load_columns,
    )

    task2 = PythonOperator(
        task_id='load_data',
        python_callable=inference_load
    )

    task3 = PythonOperator(
        task_id='load_model',
        python_callable=load_model
    )

    task4 = PythonOperator(
        task_id="inference",
        python_callable=run_inference
    )

    task0 >> task1 >> task2 >> task3 >> task4