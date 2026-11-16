import os
import sys
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DB_DIR = "chroma_db"
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")


def get_db():
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


def ingest_pdf(path):
    """Load a PDF, split into chunks, embed and store in ChromaDB."""
    name = os.path.basename(path)
    docs = PyMuPDFLoader(path).load()
    for d in docs:
        d.metadata["source"] = name

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150
    ).split_documents(docs)

    db = get_db()
    # remove old chunks of the same file so re-uploading does not create duplicates
    db._collection.delete(where={"source": name})
    db.add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "docs"
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(".pdf"):
            n = ingest_pdf(os.path.join(folder, f))
            print(f"{f} -> {n} chunks")
    print("Ingestion finished.")