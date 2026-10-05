from main import supabase


PROCEDURES_BUCKET = "bank-procedures"


def load_procedure(filename: str) -> str:
    """Télécharge une procédure Markdown depuis Supabase Storage."""
    file_bytes = (
        supabase.storage
        .from_(PROCEDURES_BUCKET)
        .download(filename)
    )

    return file_bytes.decode("utf-8")


if __name__ == "__main__":
    content = load_procedure("atm_cash_not_received.md")

    print("Procédure récupérée depuis Supabase :\n")
    print(content)