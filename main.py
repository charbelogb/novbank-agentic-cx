import sqlite3
from pathlib import Path
from pprint import pprint


DB_PATH = Path(__file__).parent / "novabank.db"


def get_customer_transactions(customer_id: str) -> list[dict]:
    """Consulte les transactions du client dans la base bancaire fictive."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Base absente. Exécute d'abord : python init_db.py"
        )

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        rows = connection.execute(
            """
            SELECT
                transaction_id,
                transaction_date,
                type,
                amount,
                currency,
                status,
                description
            FROM transactions
            WHERE customer_id = ?
            ORDER BY transaction_date DESC, transaction_id
            """,
            (customer_id,),
        ).fetchall()

        return [dict(row) for row in rows]
    finally:
        connection.close()


if __name__ == "__main__":
    current_customer_id = "CUST-001"
    transactions = get_customer_transactions(current_customer_id)
    pprint(transactions, sort_dicts=False)