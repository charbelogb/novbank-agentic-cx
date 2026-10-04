import sqlite3
from datetime import date, timedelta
from pathlib import Path


DB_PATH = Path(__file__).parent / "novabank.db"


def init_database() -> None:
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    transactions = [
        (
            "TXN-001", "CUST-001", yesterday,
            "ATM_WITHDRAWAL", 100_000, "XOF", "DEBITED",
            "Retrait au distributeur — Cotonou Centre",
        ),
        (
            "TXN-002", "CUST-001", yesterday,
            "CARD_PAYMENT", 15_000, "XOF", "DEBITED",
            "Paiement au supermarché",
        ),
        (
            "TXN-003", "CUST-002", yesterday,
            "ATM_WITHDRAWAL", 100_000, "XOF", "DEBITED",
            "Retrait au distributeur — Porto-Novo",
        ),
    ]

    connection = sqlite3.connect(DB_PATH)

    try:
        with connection:
            connection.execute("""
                               CREATE TABLE IF NOT EXISTS transactions (
                                                                           transaction_id TEXT PRIMARY KEY,
                                                                           customer_id TEXT NOT NULL,
                                                                           transaction_date TEXT NOT NULL,
                                                                           type TEXT NOT NULL,
                                                                           amount INTEGER NOT NULL,
                                                                           currency TEXT NOT NULL,
                                                                           status TEXT NOT NULL,
                                                                           description TEXT NOT NULL
                               )
                               """)

            connection.executemany("""
                INSERT OR REPLACE INTO transactions (
                    transaction_id,
                    customer_id,
                    transaction_date,
                    type,
                    amount,
                    currency,
                    status,
                    description
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, transactions)
    finally:
        connection.close()

    print(f"Base de démonstration prête : {DB_PATH}")


if __name__ == "__main__":
    init_database()