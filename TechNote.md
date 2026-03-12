# Tech Note: FinAnalyzer Pro Architecture and Development Insights

## Team Members and Roles

- **Yuhui Zhang** — Project Lead and Architecture Design. Designed the agentic query expansion workflow and overall prompt strategy.
- **Haitong Huang** — Vector Database and Data Engineering. Managed FAISS local persistence and PDF chunking.
- **Shangjun Zhang** — Machine Learning Engineer. Evaluated and compared Ollama, HuggingFace, and Gemini embedding models.
- **Tieyuan Qian** — Financial Domain Specialist. Analyzed GAAP reporting inconsistencies and helped define the Universal Accounting Protocols.
- **Yixin Wei** — QA and Testing. Developed adversarial test questions and evaluated system failure cases.
- **Ziyu Wang** — UI/UX and Deployment. Built the Streamlit interface and handled Auditor Console state management.

---

## 1. Approach and Architecture

FinAnalyzer Pro is a Retrieval-Augmented Generation (RAG) system designed to support the analysis of complex and non-standardized SEC 10-K filings.

Rather than using a simple linear RAG pipeline, the system adopts an **agentic RAG architecture**. When a user submits a query, the system first routes it through an LLM-based expansion step instead of directly retrieving documents. This step rewrites the original question into four or more financially relevant search expressions. For example, a query containing “R&D” may also be expanded into terms such as “technology and infrastructure” or other reporting variants used in company filings.

These expanded queries are then executed across the vector database in parallel. The retrieved results are merged, deduplicated, and passed into the final reasoning stage. This design improves recall when companies use inconsistent terminology or omit standard line items from their statements.

## 2. Model Choice and Prompt Strategy

- **Reasoning Model:** We selected **Gemini 2.5 Flash** as the primary reasoning model because it offers fast response time and a large context window, both of which are useful when processing long financial filings and multiple dense tables.
- **Prompt Strategy:** Instead of relying on company-specific rules, we designed a generalized framework called **Universal Accounting Protocols**. This prompt instructs the model to reason across financial statements and perform **zero-shot calculation** when a target metric is not explicitly reported.

For example, Microsoft does not always present “Total Operating Expenses” as a single line item in the income statement. In such cases, the prompt directs the model to identify the relevant components, such as Cost of Revenue, Research and Development, and Selling, General, and Administrative Expenses, and then calculate the total explicitly.

## 3. RAG Configuration

- **Embedding Model:** During development, we tested local options such as Ollama `nomic-embed` and HuggingFace `BGE-Small`, but ultimately adopted **`models/gemini-embedding-001`**. In our experiments, this model produced stronger semantic retrieval across varied accounting language and multi-page filing contexts.
- **Chunking Strategy:** We used a `RecursiveCharacterTextSplitter` with a **chunk size of 2500** and a **chunk overlap of 600**.
- **Vector Store:** We implemented **FAISS with local persistence** using `save_local` and `load_local`. This allowed the vector index to be stored on disk and reused across sessions.

This persistence layer significantly improved efficiency. Instead of re-embedding 100+ page filings every time the application restarted, the system could reload the saved vector store almost instantly, greatly reducing both API calls and startup latency.

---

## 4. Insights Gained and Failed Approaches

Several important design decisions came from failed early attempts.

### Failed Approach 1: Small Chunk Sizes

Our initial chunk size was 500. This often split large financial tables across multiple chunks, separating row values from their corresponding headers, years, or statement titles. As a result, the model sometimes received incomplete table fragments and produced incorrect or hallucinated answers. Increasing the chunk size to 2500 improved table integrity and led to more reliable retrieval.

### Failed Approach 2: Standard Retrieval for Missing Line Items

A standard semantic retrieval pipeline performed poorly when the user asked for metrics that were not explicitly written in the filing. For example, when querying Microsoft’s Total Operating Expenses, the model initially returned “DATA NOT RETRIEVED” because that exact phrase was absent from the source text.

This revealed a key limitation: semantic search alone cannot recover values that must be inferred from component items. To address this, we introduced agentic query expansion so the system could search for the underlying components of a metric rather than only the metric name itself.

### Failed Approach 3: Model Deprecation and API Instability

During development, we encountered repeated HTTP 400 and 404 errors related to deprecated Google embedding model names, including older versions such as `models/embedding-001`. This highlighted an important MLOps issue: external model APIs may change without much warning. We resolved the problem by migrating to the stable `gemini-embedding-001` endpoint and updating the embedding pipeline accordingly.

---

## 5. Assessment of System Strengths and Weaknesses

### Strengths

- **Terminology Resolution:** The system performs well when different companies use different labels for similar accounting concepts. For example, it can recognize that Amazon’s “Technology and infrastructure” may correspond to an R&D-related category.
- **Contextual Synthesis:** By combining multiple expanded retrieval paths, the system can cross-reference information from several statements, such as the balance sheet and cash flow statement, before producing an answer. This reduces the chance of missing relevant evidence.

### Weaknesses and Boundaries

- **Arithmetic Reliability:** Although the prompt encourages explicit calculation, LLMs are still language models rather than precise calculators. In longer multi-line additions, the model may occasionally make small arithmetic mistakes.
- **Context Window Dilution:** Higher retrieval depth increases the chance of capturing relevant evidence, but it also introduces more noise. When too many dense financial chunks are included in the prompt, important figures may be overlooked because they are buried inside a large context window.

---

## 6. Conclusion

FinAnalyzer Pro demonstrates that financial-document QA requires more than basic RAG. In the context of SEC 10-K analysis, inconsistent terminology, omitted line items, and dense table structures create retrieval challenges that a standard pipeline cannot reliably solve.

Our development process showed that stronger performance came from three combined choices: agentic query expansion, a generalized financial reasoning prompt, and a persistent vector database architecture. At the same time, our experiments also clarified the current boundaries of LLM-based financial auditing, especially in arithmetic precision and long-context reasoning.

Overall, the project provided both a functional prototype and a set of practical engineering lessons about retrieval design, prompt strategy, and model operations in high-stakes financial NLP tasks.
