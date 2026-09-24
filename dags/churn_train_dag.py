import os
import shutil
import logging
import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.decorators import task
from config.settings import (
    DAG_ID,
    DAG_START_DATE,
    DAG_SCHEDULE,
    DAG_CATCHUP,
    TRAIN_RAW_FILE,
    TRAIN_PREPROCESSED_FILE,
    TEST_FILE,
    TRAIN_SPLIT_FILE,
    ACCURACY_THRESHOLD,
    TRIGGER_RULE_REGISTRY,
    REAL_DATA,
    AUDITED_FILE,
    DATA_DIR,
)
from pipeline.database import save_metrics
from pipeline.data_loader import (load_training_data, load_real_data, load_audited_data, load_model)
from pipeline.evaluate import run_evaluation, process_performance_check
from pipeline.model_registry import model_registry, decide_model, columns_registry
from pipeline.preprocess import preprocess_data, split_data
from pipeline.train import run_training
from pipeline.utils import sensor, consolidate_train_data

logger = logging.getLogger("airflow.task")

def cleanup():
    """Removes existing local data directory and recreates an empty one."""
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
        os.makedirs(DATA_DIR, exist_ok=True)


with DAG(
        dag_id=DAG_ID,
        start_date=DAG_START_DATE,
        schedule=DAG_SCHEDULE,
        catchup=DAG_CATCHUP,
) as dag:

    task_cleanup = PythonOperator(
        task_id="task_cleanup",
        python_callable=cleanup
    )


    @task.branch
    def execution_mode():
        """Determines whether to execute initial training or evaluation on audited data."""
        return sensor()

    @task
    def train_pipeline():
        """Loads raw training data, preprocesses, splits, and executes model training."""
        load_training_data()
        preprocess_data(TRAIN_RAW_FILE, TRAIN_PREPROCESSED_FILE)
        split_data(TRAIN_PREPROCESSED_FILE, TRAIN_SPLIT_FILE, TEST_FILE)
        run_training()
        return "train_done"


    @task
    def evaluate_train_pipeline(train_status):
        """Evaluates trained models against real validation data and stores metrics."""
        load_real_data()
        results = run_evaluation(REAL_DATA)
        save_metrics(results)
        return results


    @task
    def evaluate_audited_pipeline():
        """Evaluates production model against audited production data and stores metrics."""
        load_audited_data()
        load_model()
        results = run_evaluation(AUDITED_FILE)
        save_metrics(results)
        return results


    @task.branch(trigger_rule="none_failed_min_one_success")
    def check_performance(ti=None):
        """Checks model evaluation metrics against defined accuracy threshold."""
        results = ti.xcom_pull(task_ids="evaluate_audited_pipeline")
        if not results:
            results = ti.xcom_pull(task_ids="evaluate_train_pipeline")
        return process_performance_check(results, ACCURACY_THRESHOLD)


    @task
    def low_performance_alert():
        """Logs a warning when performance drops below threshold and triggers retraining."""
        logger.warning("Performance below acceptable threshold! Starting retraining with expanded dataset...")
        return "start_retrain"


    @task
    def expand_and_retrain_pipeline(retrain_status):
        """Consolidates audited data with original dataset and retrains all candidate models."""
        logger.info("Running preprocessing and training on expanded base...")
        TRAIN_PATH = load_training_data()
        AUDITED_PATH = load_audited_data()
        consolidate_train_data(TRAIN_PATH, AUDITED_PATH, TRAIN_PREPROCESSED_FILE)
        split_data(TRAIN_PREPROCESSED_FILE, TRAIN_SPLIT_FILE, TEST_FILE)
        run_training()
        return "retrain_done"

    @task
    def stable_performance():
        """Logs confirmation when model performance meets requirements and skips retraining."""
        logger.info("Model performing well. Skipping retraining stage.")
        return "skip_retrain"


    @task(trigger_rule=TRIGGER_RULE_REGISTRY)
    def registry_pipeline(ti=None):
        """Selects best model based on evaluation metrics and registers artifacts to S3."""
        results = ti.xcom_pull(task_ids="evaluate_audited_pipeline")
        if not results:
            results = ti.xcom_pull(task_ids="evaluate_train_pipeline")
        best = decide_model(results)
        model_registry(best)
        columns_registry()


    # ==============================================================================
    # PIPELINE TASK FLOW DEFINITION
    # ==============================================================================

    # 1. Execution Mode Entry Point
    mode = execution_mode()

    # 2. Branch A: Initial Training Path
    train_status = train_pipeline()
    train_eval_results = evaluate_train_pipeline(train_status)

    # 3. Branch B: Audited Data Path (Monitoring Evaluation)
    audited_eval_results = evaluate_audited_pipeline()

    # 4. Decision Funnel & Performance Assessment
    decision = check_performance()
    low_perf = low_performance_alert()
    retrain = expand_and_retrain_pipeline(low_perf)
    stable_perf = stable_performance()

    # 5. Model Registry Task
    registry = registry_pipeline()

    # ==============================================================================
    # TASK DEPENDENCIES & BRANCHING LOGIC
    # ==============================================================================

    # Setup & Execution Mode
    task_cleanup >> mode

    # Branching: Decision based on execution mode
    mode >> train_status >> train_eval_results >> decision
    mode >> audited_eval_results >> decision

    # Conditional Paths: Based on performance check
    decision >> low_perf >> retrain >> registry
    decision >> stable_perf >> registry