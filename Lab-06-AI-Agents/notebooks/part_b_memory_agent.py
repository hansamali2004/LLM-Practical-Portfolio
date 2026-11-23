import datetime
import json
import re
import sqlite3
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

MODEL = "llama3.2:3b"
DB_PATH = Path(__file__).with_name("memory.db")

SYSTEM = ("You are an intelligent student assistant. Use search_student_memory when the user asks "
          "about themselves or things they told you earlier. Use the calculator for maths. "
          "Never guess personal facts. Answer briefly.")


# ---------------- LONG-TERM MEMORY (SQLite) ----------------
def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS memories(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT UNIQUE,
            created TEXT)""")


def save_fact(fact):
    with sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT OR IGNORE INTO memories(fact, created) VALUES (?, ?)",
                  (fact, datetime.datetime.now().isoformat(timespec="seconds")))


def all_facts():
    with sqlite3.connect(DB_PATH) as c:
        return [r[0] for r in c.execute("SELECT fact FROM memories ORDER BY id DESC")]


extractor = ChatOllama(model=MODEL, temperature=0, format="json")


def extract_facts(text):
    """Ask the LLM to pull out personal facts the user states about themselves."""
    prompt = ('Extract personal facts the user states about themselves (name, degree, interests, '
              'preferences, goals). Reply ONLY with JSON like {"facts": ["The user\'s name is Alex"]}. '
              'If there are none reply {"facts": []}.\n\nUser message: ' + text)
    try:
        data = json.loads(extractor.invoke(prompt).content)
        return [f for f in data.get("facts", []) if isinstance(f, str) and f.strip()]
    except Exception:
        return []


# ---------------- TOOLS ----------------
@tool
def calculator(expression: str) -> str:
    """Calculate a math expression such as '12*(3+4)'."""
    expr = expression.replace("×", "*").replace("x", "*").replace("^", "**")
    if not re.fullmatch(r"[0-9+\-*/().%\s]+", expr):
        return "Error: invalid expression."
    try:
        return f"{expression} = {eval(expr, {'__builtins__': {}}, {})}"
    except Exception as e:
        return f"Error: {e}"


@tool
def search_student_memory(query: str) -> str:
    """Search previously stored information about the user (name, degree, interests, etc.)."""
    facts = all_facts()
    if not facts:
        return "No matching memories found."
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2]
    hits = [f for f in facts if any(w in f.lower() for w in words)]
    return "\n".join(hits or facts[:10])


# ---------------- AGENT LOOP ----------------
def chat_loop(tool_list, title, system=SYSTEM):
    init_db()
    tools = {t.name: t for t in tool_list}
    llm = ChatOllama(model=MODEL, temperature=0).bind_tools(tool_list)
    history = [SystemMessage(system)]  # short-term memory (this conversation)

    print("=" * 40, f"\n{title}\nSHORT-TERM MEMORY\nLONG-TERM SQLITE MEMORY\n(type exit to quit)\n", "=" * 40)
    while True:
        user = input("\nYou: ").strip()
        if user.lower() in ("exit", "quit"):
            break

        new = extract_facts(user)
        if new:
            print("\n[NEW MEMORIES DETECTED]")
            for f in new:
                save_fact(f)
                print("[SAVED]", f)
        else:
            print("\n[MEMORY] No new information to save.")

        history.append(HumanMessage(user))
        answer = "Stopped: too many steps."
        for _ in range(5):
            reply = llm.invoke(history)
            history.append(reply)
            if not reply.tool_calls:
                answer = reply.content
                break
            for call in reply.tool_calls:
                print(f"\n[TOOL SELECTED] {call['name']}")
                print(f"[TOOL ARGUMENTS] {call['args']}")
                result = str(tools[call["name"]].invoke(call["args"]))
                print(f"[TOOL RESULT] {result[:500]}")
                history.append(ToolMessage(result, tool_call_id=call["id"]))
        print("\n[SHORT-TERM MEMORY] Conversation updated.")
        print("\nAgent:", answer)


if __name__ == "__main__":
    chat_loop([calculator, search_student_memory], "INTELLIGENT STUDENT AI AGENT")