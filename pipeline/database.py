import logging
from airflow.providers.postgres.hooks.postgres import PostgresHook
from config.settings import CONN, TABLE_NAME, DB_COLUMNS

logger = logging.getLogger(__name__)


def save_metrics(results):
    """Inserts model evaluation metrics into the configured PostgreSQL database table."""
    hook = PostgresHook(postgres_conn_id=CONN)
    conn = hook.get_conn()
    cursor = conn.cursor()

    placeholders = ", ".join(["%s"] * len(DB_COLUMNS))
    columns_str = ", ".join(DB_COLUMNS)
    sql = f"INSERT INTO {TABLE_NAME} ({columns_str}) VALUES ({placeholders})"

    for result in results:
        row = [result[col] for col in DB_COLUMNS]
        cursor.execute(sql, row)
        logger.info(
            "\033[1;32m[SUCCESS] Metrics successfully inserted into table: %s | Model: %s\033[0m",
            TABLE_NAME,
            result.get("model_name"),
        )

    conn.commit()
    cursor.close()
    conn.close()