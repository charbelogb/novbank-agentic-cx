import os

from main import supabase
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings

PROCEDURES_BUCKET = "bank-procedures"


def load_procedure(filename: str) -> str:
    """Télécharge une procédure Markdown depuis Supabase Storage."""
    file_bytes = (
        supabase.storage
        .from_(PROCEDURES_BUCKET)
        .download(filename)
    )

    return file_bytes.decode("utf-8")

def split_procedure(content: str, filename: str) -> list[Document]:
    """Découpe une procédure Markdown selon ses titres."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),
            ("##", "section"),
        ],
        strip_headers=False,
    )

    chunks = splitter.split_text(content)

    for index, chunk in enumerate(chunks):
        chunk.metadata["source"] = filename
        chunk.metadata["bucket"] = PROCEDURES_BUCKET
        chunk.metadata["chunk_index"] = index

    return chunks

def embed_chunks(chunks: list[Document]) -> list[list[float]]:
    """Calcule un embedding pour chaque passage."""
    if not chunks:
        return []

    embeddings = OpenAIEmbeddings(
        model=os.environ["OPENAI_EMBEDDING_MODEL"],
    )

    texts = [chunk.page_content for chunk in chunks]

    return embeddings.embed_documents(texts)

if __name__ == "__main__":
    filename = "atm_cash_not_received.md"

    content = load_procedure(filename)
    chunks = split_procedure(content, filename)
    vectors = embed_chunks(chunks)

    print(f"Nombre de passages : {len(chunks)}")
    print(f"Nombre de vecteurs : {len(vectors)}")

    if vectors:
        print(f"Dimensions du premier vecteur : {len(vectors[0])}")
        print(f"Ses 5 premières valeurs : {vectors[0][:5]}")