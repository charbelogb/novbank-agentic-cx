import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import URL

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.retrievers import VectorIndexRetriever

from main import supabase


load_dotenv(Path(__file__).parent / ".env")

PROCEDURES_BUCKET = "bank-procedures"
EMBEDDING_DIMENSIONS = 1536


def load_procedure(filename: str) -> Document:
    """Télécharge la procédure et crée un document LlamaIndex."""
    file_bytes = (
        supabase.storage
        .from_(PROCEDURES_BUCKET)
        .download(filename)
    )

    return Document(
        text=file_bytes.decode("utf-8"),
        id_=f"{PROCEDURES_BUCKET}/{filename}",
        metadata={
            "source": filename,
            "bucket": PROCEDURES_BUCKET,
        },
    )


def get_vector_store() -> PGVectorStore:
    """Configure le stockage PostgreSQL utilisé par LlamaIndex."""
    connection = URL.create(
        drivername="postgresql+psycopg2",
        username=os.environ["SUPABASE_DB_USER"],
        password=os.environ["SUPABASE_DB_PASSWORD"],
        host=os.environ["SUPABASE_DB_HOST"],
        port=int(os.environ["SUPABASE_DB_PORT"]),
        database=os.environ["SUPABASE_DB_NAME"],
        query={"sslmode": "require"},
    )

    async_connection = connection.set(
        drivername="postgresql+asyncpg",
        query={"ssl": "require"},
    )

    return PGVectorStore(
        connection_string=connection.render_as_string(
            hide_password=False
        ),
        async_connection_string=async_connection.render_as_string(
            hide_password=False
        ),
        schema_name="rag",
        table_name="bank_procedures",
        embed_dim=EMBEDDING_DIMENSIONS,
        initialization_fail_on_error=True,
    )


def index_procedure(filename: str) -> None:
    """Découpe, vectorise et enregistre une procédure."""
    document = load_procedure(filename)

    embeddings = OpenAIEmbedding(
        model=os.environ["OPENAI_EMBEDDING_MODEL"],
        dimensions=EMBEDDING_DIMENSIONS,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=get_vector_store()
    )

    VectorStoreIndex.from_documents(
        [document],
        storage_context=storage_context,
        embed_model=embeddings,
        transformations=[
            SentenceSplitter(
                chunk_size=512,
                chunk_overlap=64,
            )
        ],
        show_progress=True,
    )

    print(f"Procédure indexée : {filename}")

def get_retriever() -> VectorIndexRetriever:
    """Prépare la recherche dans les passages déjà indexés."""
    embeddings = OpenAIEmbedding(
        model=os.environ["OPENAI_EMBEDDING_MODEL"],
        dimensions=EMBEDDING_DIMENSIONS,
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=get_vector_store(),
        embed_model=embeddings,
    )

    return index.as_retriever(similarity_top_k=2)

if __name__ == "__main__":
    index_procedure("atm_cash_not_received.md")