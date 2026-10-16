import os
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv
from supabase import Client, create_client


load_dotenv(Path(__file__).parent / ".env")

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SECRET_KEY")

if not supabase_url or not supabase_key:
    raise RuntimeError(
        "SUPABASE_URL et SUPABASE_SECRET_KEY doivent être "
        "renseignées dans le fichier .env."
    )

supabase: Client = create_client(supabase_url, supabase_key)


def get_customer_transactions(customer_id: str) -> list[dict]:
    """Consulte les transactions du client dans Supabase."""
    if not customer_id or not customer_id.strip():
        raise ValueError("Le customer_id est obligatoire.")

    response = (
        supabase.table("transactions")
        .select(
            "transaction_id, transaction_date, type, "
            "amount, currency, status, description"
        )
        .eq("customer_id", customer_id)
        .order("transaction_date", desc=True)
        .order("transaction_id")
        .execute()
    )

    return response.data


if __name__ == "__main__":
    transactions = get_customer_transactions("CUST-001")
    pprint(transactions, sort_dicts=False)