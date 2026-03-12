import os
import tempfile
import time
import shutil
import json
import streamlit as st

# Resolve potential OpenMP conflicts on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Streamlit page config
st.set_page_config(page_title="FinAnalyzer Pro (Ollama Local)", page_icon="🏦", layout="wide")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

# --- CONFIGURATION ---
DB_DIR = "faiss_index_store"
METADATA_FILE = os.path.join(DB_DIR, "indexed_files.json")
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"  
OLLAMA_LLM_MODEL = "llama3.1"

# --- PERSISTENCE HELPERS ---
def save_metadata(file_names):
    if not os.path.exists(DB_DIR): os.makedirs(DB_DIR)
    with open(METADATA_FILE, "w") as f: json.dump(list(file_names), f)

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f: return json.load(f)
    return []

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Admin Console")
    if st.button("💬 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown("### 🗄️ Knowledge Base")
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = load_metadata()
    if st.session_state.processed_files:
        st.success(f"Index loaded: {len(st.session_state.processed_files)} files")
        for f in st.session_state.processed_files: st.caption(f"✅ {f}")
    else: st.info("No documents in memory.")
    st.divider()
    if st.button("🔥 Wipe Database"):
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
            for key in ["vector_store", "processed_files", "messages"]:
                if key in st.session_state: del st.session_state[key]
            st.success("System Reset!")
            time.sleep(1)
            st.rerun()

# --- MODELS ---
@st.cache_resource
def init_models():
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    # Temperature 0.0 for maximum financial precision
    llm = OllamaLLM(model=OLLAMA_LLM_MODEL, temperature=0.0)
    return embeddings, llm

embeddings, llm = init_models()

# --- VECTOR STORE INIT ---
if "vector_store" not in st.session_state and os.path.exists(os.path.join(DB_DIR, "index.faiss")):
    try:
        st.session_state.vector_store = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    except: pass

# --- UPLOAD ---
uploaded_files = st.file_uploader("Upload 10-K PDFs", accept_multiple_files=True, type=["pdf"])
if uploaded_files:
    current_files = sorted([f.name for f in uploaded_files])
    if st.session_state.get("processed_files") != current_files:
        with st.spinner("Indexing documents..."):
            all_docs = []
            with tempfile.TemporaryDirectory() as tmp:
                for f in uploaded_files:
                    path = os.path.join(tmp, f.name)
                    with open(path, "wb") as b: b.write(f.getbuffer())
                    all_docs.extend(PyPDFLoader(path).load())
            docs = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250).split_documents(all_docs)
            st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
            st.session_state.vector_store.save_local(DB_DIR)
            save_metadata(current_files)
            st.session_state.processed_files = current_files
        st.success("Knowledge Base Ready!")

# --- CHAT UI ---
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if user_input := st.chat_input("Compare R&D, Revenue, or Net Income across companies..."):
    if "vector_store" not in st.session_state:
        st.warning("Please upload PDFs first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    # --- BALANCED ENTITY-AWARE RETRIEVAL ---
    with st.spinner("Analyzing cross-company data layers..."):
        balanced_docs = []
        # Detection logic
        entities = []
        if "amazon" in user_input.lower(): entities.append("Amazon")
        if any(kw in user_input.lower() for kw in ["google", "alphabet"]): entities.append("Alphabet")
        if any(kw in user_input.lower() for kw in ["microsoft", "msft"]): entities.append("Microsoft")

        if entities:
            # Targeted retrieval: Ensure each company gets its own slots
            for ent in entities:
                targeted_query = f"{ent} 2024 financial expenses, R&D, and net income: {user_input}"
                co_docs = st.session_state.vector_store.similarity_search(targeted_query, k=5)
                balanced_docs.extend(co_docs)
        else:
            # Fallback to general MMR for non-specific queries
            balanced_docs = st.session_state.vector_store.max_marginal_relevance_search(user_input, k=10)

        # Build History Context
        history_ctx = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-4:-1]])
        
        # TABLE ENFORCER PROMPT
        template = f"""
        You are a Senior Financial Auditor. Your task is to provide a balanced comparison.
        
        STRICT PROTOCOL:
        1. FOR COMPARISONS: You MUST output a Markdown table with columns: [Company], [Key Metric Value], [Context/Source].
        2. You MUST search your context for ALL companies mentioned in the question (Amazon, Alphabet, Microsoft).
        3. If data for a company is not in the provided context, fill that row with "DATA NOT FOUND IN RETRIEVED SEGMENTS".
        4. Cite the exact page number for every figure.

        Chat History:
        {history_ctx}

        Context Information:
        {{context}}

        User Question: {{question}}
        Professional Table and Analysis:"""

        prompt_obj = PromptTemplate(input_variables=["context", "question"], template=template)

        # Execute Chain
        qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt_obj)
        response = qa_chain.invoke({
            "input_documents": balanced_docs, 
            "question": user_input
        })
        
        answer = response["output_text"]

        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("🔍 Balanced Data Sources (Audit Trail)"):
                for doc in balanced_docs:
                    src = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    pg = doc.metadata.get('page', '?')
                    st.caption(f"Source: {src} | Page: {pg}")
                    st.write(doc.page_content[:300] + "...")
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})