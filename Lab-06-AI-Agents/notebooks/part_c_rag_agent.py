from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings

from part_b_memory_agent import calculator, chat_loop, search_student_memory

# Reuse the vector DB built in Lab 5
RAG_DB = Path(__file__).resolve().parents[2] / "Lab-05-RAG-Pipeline" / "chroma_db"
db = Chroma(persist_directory=str(RAG_DB), embedding_function=OllamaEmbeddings(model="qwen3-embedding:0.6b"))

SYSTEM = ("You are an intelligent university assistant with three tools. "
          "Use search_university_knowledge for ANY question about the university, degree programme, "
          "modules, grading, fees, attendance or services. "
          "Use search_student_memory for facts the user told you about themselves. "
          "Use calculator for maths. Answer only from tool results; if the information is not found, say so. "
          "Answer briefly.")


@tool
def search_university_knowledge(query: str) -> str:
    """Search the university knowledge base (handbook PDFs) for relevant information."""
    print(f"\n[EXECUTING] Searching university knowledge for: '{query}'...")
    docs = db.similarity_search(query, k=4)
    if not docs:
        return "No relevant information found."
    return "\n\n".join(
        f"Source: {d.metadata.get('source')}, Page: {d.metadata.get('page', 0) + 1}\n{d.page_content}"
        for d in docs
    )


if __name__ == "__main__":
    chat_loop([calculator, search_student_memory, search_university_knowledge],
              "RAG-POWERED STUDENT AI AGENT", system=SYSTEM)