# Lab 06 – Tool-Using AI Agents with Ollama

## Objective
Build AI agents that use tools, memory and RAG with a local LLM (llama3.2:3b via Ollama).

## Tools
Python, LangChain, Ollama, SQLite, ChromaDB

## Part A – Calculator agent + multiple tools
Tools: calculator (opens Calculator app), date/time, text editor (TextEdit), file search, file creation, system info.
![calc](screenshots/01_calculator_agent.png)
![tools](screenshots/02_multi_tools.png)

## Part B – Memory (SQLite)
Short-term memory = conversation history. Long-term memory = SQLite table of facts extracted by the LLM.
![memory](screenshots/03_memory_agent.png)

## Part C – RAG agent
Agent with calculator, memory search and university knowledge search (ChromaDB from Lab 5).
![rag](screenshots/05_rag_agent.png)

## Observations
- Correct tool choices:
- Wrong tool calls / hallucinations (small 3B model):
- How memory improved answers:

## Adaptations for macOS
Windows Calculator/Notepad replaced with macOS Calculator/TextEdit; pyautogui replaced with AppleScript.

## How to run
source ../Lab-05-RAG-Pipeline/rag_env/bin/activate
cd notebooks
python part_a_agent.py
python part_b_memory_agent.py
python part_c_rag_agent.py