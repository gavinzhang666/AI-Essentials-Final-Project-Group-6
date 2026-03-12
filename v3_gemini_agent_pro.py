import os
import tempfile
import time
import shutil
import json
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# --- System Configuration ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv(find_dotenv(), override=True)

raw_key = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = raw_key.strip().strip('"').strip("'") if raw_key else None

st.set_page_config(page_title="FinAnalyzer Pro", page_icon="🏦", layout="wide")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

try:
    from langchain.chains.question_answering import load_qa_chain
except ImportError:
    from langchain_community.chains.question_answering import load_qa_chain

# --- Storage Configuration ---
DB_DIR = "faiss_index_universal"
METADATA_FILE = os.path.join(DB_DIR, "indexed_files.json")

def save_metadata(file_names):
    if not os.path.exists(DB_DIR): os.makedirs(DB_DIR)
    with open(METADATA_FILE, "w") as f: json.dump(list(file_names), f)

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f: return json.load(f)
    return []

# --- Model Initialization ---
@st.cache_resource
def init_ai(api_key):
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=api_key,
        task_type="retrieval_document"
    )
    # Gemini 2.5 Flash Possesses exceptional ability to find a needle in a haystack within lengthy texts
    llm_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.0, 
        google_api_key=api_key
    )
    return embed_model, llm_model

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY is missing.")
    st.stop()

embeddings, llm = init_ai(GOOGLE_API_KEY)

# --- Sidebar: Management and Persistence  ---
with st.sidebar:
    st.title("🛡️ Auditor Console")
    st.success("🔑 System Active")
    
    if st.button("💬 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = load_metadata()
    
    if st.session_state.processed_files:
        st.success(f"Indexed: {len(st.session_state.processed_files)} files")
        for f in st.session_state.processed_files: st.caption(f"✅ {f}")
    
    if st.button("🔥 Wipe Local Index"):
        if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)
        for key in ["vector_store", "processed_files", "messages"]:
            st.session_state.pop(key, None)
        st.rerun()

# --- Automatically restore local indexes ---
if "vector_store" not in st.session_state and os.path.exists(os.path.join(DB_DIR, "index.faiss")):
    try:
        st.session_state.vector_store = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    except: pass

# --- File Upload ---
st.title("📊 Financial RAG (Full Gemini Cloud Agentic)")
uploaded_files = st.file_uploader("Upload 10-K PDFs", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    f_names = sorted([f.name for f in uploaded_files])
    if st.session_state.get("processed_files") != f_names:
        with st.spinner("Embedding PDFs (High-Capacity Chunks)..."):
            all_docs = []
            with tempfile.TemporaryDirectory() as tmp:
                for f in uploaded_files:
                    path = os.path.join(tmp, f.name)
                    with open(path, "wb") as b: b.write(f.getbuffer())
                    all_docs.extend(PyPDFLoader(path).load())
            
            # CRITICAL FIX: Increase Chunk Size to 2500 to prevent Microsoft's entire income statement from being severed.
            splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=600)
            docs = splitter.split_documents(all_docs)
            st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
            st.session_state.vector_store.save_local(DB_DIR)
            save_metadata(f_names)
            st.session_state.processed_files = f_names
        st.success("✅ Universal Index Built.")

# --- Chat and Agentic Retrieval ---
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if user_input := st.chat_input("Ask any complex financial question..."):
    if "vector_store" not in st.session_state:
        st.error("Upload PDFs first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    # ==========================================
    # STEP 1: DYNAMIC QUERY EXPANSION (Agentic Search)
    # ==========================================
    with st.spinner("Agent thinking: Expanding search terminology..."):
        expansion_template = """
        You are a financial search expert. 
        Analyze this question: "{question}"
        
        Generate exactly 4 distinct, highly targeted search queries to ensure we don't miss data.
        CRITICAL RULES:
        1. Include queries targeting exact statement names (e.g., "INCOME STATEMENTS", "Consolidated Statements of Operations").
        2. If the user asks for 'Total Operating Expenses', generate a query that specifically looks for constituent parts: "Cost of revenue Research and development Sales and marketing General and administrative".
        
        Return ONLY a comma-separated list of the 4 queries, nothing else.
        """
        expansion_prompt = PromptTemplate(template=expansion_template, input_variables=["question"])
        
        try:
            expanded_queries_str = llm.invoke(expansion_prompt.format(question=user_input)).content
            search_queries = [q.strip() for q in expanded_queries_str.split(',')]
            search_queries.append(user_input) # Always include the original question as a fallback search query
            
            st.sidebar.markdown("### 🧠 Agent's Search Strategy")
            for i, q in enumerate(search_queries):
                st.sidebar.caption(f"Search {i+1}: {q}")
                
        except Exception as e:
            search_queries = [user_input]

    # ==========================================
    # STEP 2: ENSEMBLE RETRIEVAL (Large-Capacity Recall)
    # ==========================================
    with st.spinner("Executing expanded searches across the database..."):
        raw_results = []
        # CRITICAL FIX: k=8 to ensure we get enough relevant chunks for complex queries, especially for Microsoft which has a more fragmented expense structure.
        for query in search_queries:
            raw_results.extend(st.session_state.vector_store.similarity_search(query, k=8))

        # CRITICAL FIX: Deduplication step to ensure we don't overwhelm the LLM with redundant information, especially since multiple queries may retrieve overlapping chunks.
        seen = set()
        balanced_docs = []
        for d in raw_results:
            if d.page_content not in seen:
                balanced_docs.append(d)
                seen.add(d.page_content)

    # ==========================================
    # STEP 3: UNIVERSAL AUDITOR REASONING AND SYNTHESIS
    # ==========================================
    with st.spinner("Synthesizing financial data..."):
        history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-5:-1]])

        universal_template = f"""
        You are a Senior Financial Auditor synthesizing data from 10-K filings.
        
        UNIVERSAL ACCOUNTING PROTOCOLS:
        1. Terminology Map: Amazon's R&D is explicitly 'Technology and infrastructure'.
        2. MISSING TOTALS CALCULATION: If a company (like Microsoft or Alphabet) does not have a single line item for 'Total Operating Expenses', YOU MUST CALCULATE IT by summing all relevant operating expenses found on their Income Statement. 
           - For Microsoft, sum: 'Cost of revenue' + 'Research and development' + 'Sales and marketing' + 'General and administrative'.
        3. Prioritize 'Consolidated Statements of Operations' or 'INCOME STATEMENTS' for expenses.
        4. Output comparisons in a Markdown table.
        5. Cite EXACT File Names and Page Numbers. Show your calculation steps clearly.

        History: {history_str}
        Context: {{context}}
        Question: {{question}}
        Professional Answer:"""

        prompt = PromptTemplate(input_variables=["context", "question"], template=universal_template)
        qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
        
        try:
            response = qa_chain.invoke({"input_documents": balanced_docs, "question": user_input})
            answer = response["output_text"]

            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("🔍 View Retrieved Evidence"):
                    for d in balanced_docs:
                        src = os.path.basename(d.metadata.get('source', 'Unknown'))
                        pg = d.metadata.get('page', '?')
                        st.caption(f"Doc: {src} | Page: {pg}")
                        st.write(d.page_content[:300] + "...")
                        
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Analysis Error: {e}")