from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from ingest import get_db

llm = ChatOllama(model="llama3.2:3b", temperature=0)

PROMPT = ChatPromptTemplate.from_template(
"""You are a university assistant. Answer ONLY using the context below.
If the answer is not in the context, say: "I could not find this in the documents."

Context:
{context}

Question: {question}

Answer:""")


def ask(question, k=5):
    db = get_db()
    docs = db.similarity_search(question, k=k)
    if not docs:
        return "No documents have been added yet.", []

    context = "\n\n".join(
        f"[{d.metadata.get('source')} p.{d.metadata.get('page', 0) + 1}]\n{d.page_content}"
        for d in docs
    )
    answer = (PROMPT | llm).invoke({"context": context, "question": question}).content
    sources = sorted({f"{d.metadata.get('source')} (p.{d.metadata.get('page', 0) + 1})" for d in docs})
    return answer, sources


if __name__ == "__main__":
    while True:
        q = input("\nAsk a university question (or exit): ")
        if q.strip().lower() == "exit":
            break
        a, s = ask(q)
        print("\nAnswer:", a)
        print("Sources:", s)