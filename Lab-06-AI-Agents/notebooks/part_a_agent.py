import datetime
import os
import platform
import re
import subprocess
import tempfile
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


# ---------------- TOOLS ----------------
@tool
def calculator(expression: str) -> str:
    """Calculate a math expression such as '25*4'. Opens the Calculator app and enters the expression."""
    expr = expression.replace("×", "*").replace("x", "*").replace("^", "**")
    if not re.fullmatch(r"[0-9+\-*/().%\s]+", expr):
        return "Error: only numbers and + - * / ( ) are allowed."
    try:
        result = eval(expr, {"__builtins__": {}}, {})
    except Exception as e:
        return f"Error: {e}"

    try:
        print("[EXECUTING] Opening Calculator...")
        subprocess.run(["open", "-a", "Calculator"], check=False)
        time.sleep(1.5)
        keys = expr.replace(" ", "") + "="
        script = f'tell application "System Events" to keystroke "{keys}"'
        subprocess.run(["osascript", "-e", script], check=False, capture_output=True)
        print("[SUCCESS] Expression entered in Calculator.")
    except Exception as e:
        print("[WARNING] Could not control Calculator:", e)
    return f"{expression} = {result}"


@tool
def get_date_time() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")


@tool
def open_notepad(text: str) -> str:
    """Open a text editor (TextEdit) and write the given text in it."""
    path = os.path.join(tempfile.gettempdir(), "agent_note.txt")
    with open(path, "w") as f:
        f.write(text)
    subprocess.run(["open", "-a", "TextEdit", path], check=False)
    return f"Opened