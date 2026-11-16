import os
import tempfile
import streamlit as st
from ingest import ingest_pdf
from rag_pipeline import ask

st.set_page_config(page_title="University AI Assistant", page_icon="🎓")
st.title("🎓 University AI Assistant")
st.caption("Ask questions based on university documents")

# ---------- Sidebar: upload PDFs ----------
with st.sidebar:
    st.header("Upload PDFs")
    files = st.file_uploader("Add your own PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Process uploaded PDFs") and files:
        for f in files:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                with st.spinner(f"Indexing {f.name}..."):
                    n = ingest_pdf(path)
            st.success(f"{f.name}: {n} chunks added")

    st.markdown("Pre-loaded PDFs: put them in `docs/` and run `python ingest.py docs`.")

# ---------- Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("Ask your university question..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ans, src = ask(q)
        st.markdown(ans)
        if src:
            st.caption("Sources: " + ", ".join(src))
    st.session_state.messages.append({"role": "assistant", "content": ans})