import os
import tempfile
import time
import shutil
import json
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# --- SYSTEM & API SETUP ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Force load from .env regardless of terminal launch path
load_dotenv(find_dotenv(), override=True)

# Clean API Key to prevent 400 errors
raw_key = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = raw_key.strip().strip('"').strip("'") if raw_key else None

st.set_page_config(page_title="FinAnalyzer Pro 2026", page_icon="🏦", layout="wide")

# LangChain 0.3+ Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain

# --- CONFIGURATION ---
DB_DIR = "faiss_index_hybrid_v3"
METADATA_FILE = os.path.join(DB_DIR, "indexed_files.json")
# BGE-Small remains the champion for local financial retrieval
LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5" 

def save_metadata(file_names):
    if not os.path.exists(DB_DIR): os.makedirs(DB_DIR)
    with open(METADATA_FILE, "w") as f: json.dump(list(file_names), f)

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f: return json.load(f)
    return []

@st.cache_resource
def init_ai(api_key):
    # Local Embedding Inference
    embed_model = HuggingFaceEmbeddings(model_name=LOCAL_EMBED_MODEL)
    # Using Gemini 3 Flash
    llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
    return embed_model, llm_model

if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY in .env file.")
    st.stop()

embeddings, llm = init_ai(GOOGLE_API_KEY)

# --- SIDEBAR: ADMIN & PERSISTENCE ---
with st.sidebar:
    st.title("🛡️ Auditor Console")
    st.success(f"🔑 API Key Loaded ({GOOGLE_API_KEY[:4]}...)")
    
    if st.button("💬 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("### 🗄️ Memory Status")
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = load_metadata()
    
    if st.session_state.processed_files:
        st.success(f"Indexed: {len(st.session_state.processed_files)} files")
        for f in st.session_state.processed_files: st.caption(f"✅ {f}")
    
    if st.button("🔥 Wipe & Re-index"):
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
            for key in ["vector_store", "processed_files", "messages"]:
                st.session_state.pop(key, None)
            st.success("Database Wiped!")
            time.sleep(1)
            st.rerun()

# --- RECOVER FAISS ---
if "vector_store" not in st.session_state and os.path.exists(os.path.join(DB_DIR, "index.faiss")):
    try:
        st.session_state.vector_store = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    except: pass

# --- UPLOAD & AUDIT-GRADE PROCESSING ---
st.title("📊 Financial RAG Analyst (HF Embeddings + Gemini)")
uploaded_files = st.file_uploader("Upload 10-K PDFs", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    f_names = sorted([f.name for f in uploaded_files])
    if st.session_state.get("processed_files") != f_names:
        with st.spinner("Building high-precision index..."):
            all_docs = []
            with tempfile.TemporaryDirectory() as tmp:
                for f in uploaded_files:
                    path = os.path.join(tmp, f.name)
                    with open(path, "wb") as b: b.write(f.getbuffer())
                    all_docs.extend(PyPDFLoader(path).load())
            
            # Optimization: 1500 chunk + 500 overlap keeps tables intact
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
            docs = splitter.split_documents(all_docs)
            st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
            st.session_state.vector_store.save_local(DB_DIR)
            save_metadata(f_names)
            st.session_state.processed_files = f_names
        st.success("Audit Index Ready.")

# --- CHAT & ENHANCED RETRIEVAL ---
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if user_input := st.chat_input("Compare R&D, OpEx, and Net Income..."):
    if "vector_store" not in st.session_state:
        st.error("Please upload PDFs first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    # --- ADVANCED MULTI-ENTITY PROBE ---
    with st.spinner("Scanning cross-company data layers..."):
        entities = [e for e in ["Amazon", "Alphabet", "Microsoft"] if e.lower() in user_input.lower() or (e=="Alphabet" and "google" in user_input.lower())]
        
        raw_results = []
        if entities:
            for ent in entities:
                # Path A: The Income Statement Search
                q_stmt = f"{ent} 2024 Consolidated Statements of Operations table"
                # Path B: The Terminology Search (Critical for Amazon)
                if "amazon" in ent.lower():
                    q_term = f"Amazon 2024 Technology and infrastructure and Total operating expenses"
                else:
                    q_term = f"{ent} 2024 Research and Development R&D and Total operating expenses"
                
                raw_results.extend(st.session_state.vector_store.similarity_search(q_stmt, k=5))
                raw_results.extend(st.session_state.vector_store.similarity_search(q_term, k=5))
        else:
            raw_results = st.session_state.vector_store.max_marginal_relevance_search(user_input, k=15, fetch_k=50)

        # Deduplicate to maximize context efficiency
        seen = set()
        balanced_docs = []
        for d in raw_results:
            if d.page_content not in seen:
                balanced_docs.append(d)
                seen.add(d.page_content)

        history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-5:-1]])

        # FINAL AUDITOR PROMPT
        template = f"""
        You are a Senior Financial Auditor. 
        NOTE: Amazon reports R&D as 'Technology and infrastructure'.
        
        STRICT PROTOCOL:
        1. Always output a Markdown table for comparisons.
        2. Columns: [Company], [R&D/Tech Expense], [Total Operating Expenses], [% of OpEx], [Source].
        3. Search context specifically for 'Technology and infrastructure' when analyzing Amazon.
        4. Cite the EXACT Page Number and File Name.
        5. If data is missing for a company, state 'DATA NOT RETRIEVED'.

        History: {history_str}
        Context: {{context}}
        Question: {{question}}
        Answer:"""

        prompt = PromptTemplate(input_variables=["context", "question"], template=template)
        qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
        
        try:
            response = qa_chain.invoke({"input_documents": balanced_docs, "question": user_input})
            answer = response["output_text"]

            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("🔍 Source Audit Trail"):
                    for d in balanced_docs:
                        st.caption(f"Doc: {os.path.basename(d.metadata.get('source','?'))} | Page: {d.metadata.get('page','?')}")
                        st.write(d.page_content[:300] + "...")
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Analysis failed: {e}")
