# 🐥 DocChat — Multi-Agent RAG Document Q&A System

![Language](https://img.shields.io/badge/Language-Python%203.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Framework](https://img.shields.io/badge/Framework-LangGraph-FF6B35?style=flat-square)
![LLM](https://img.shields.io/badge/LLM-Llama%203.3%2070B%20%7C%20Granite%203.3-052FAD?style=flat-square&logo=ibm&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-Hybrid%20BM25%20%2B%20Vector-2E7D32?style=flat-square)
![UI](https://img.shields.io/badge/UI-Gradio-FF7C00?style=flat-square)
![Doc](https://img.shields.io/badge/Parser-Docling-6A0DAD?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

---

## 📌 Project Overview

**DocChat** is a **production-ready multi-agent RAG (Retrieval-Augmented 
Generation) document Q&A system** powered by IBM Watsonx, LangGraph, 
and Docling.

Upload any document (PDF, DOCX, TXT, MD) and ask questions — DocChat's 
**3-agent LangGraph workflow** handles relevance checking, research 
generation, and factual verification automatically, with intelligent 
re-research loops when answers need improvement.

**Domain:** RAG + Multi-Agent AI — Document Intelligence  
**LLMs:** Meta Llama 3.3 70B (Research) + IBM Granite 3.3 8B (Relevance)  
**Embeddings:** IBM Slate 125M English Retriever v2  
**Document Parser:** Docling  
**UI:** Gradio  

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Gradio Web UI (app.py)                       │
│   Upload PDF/DOCX/TXT/MD + Enter Question                │
│   Session state caching (SHA-256 file hashes)            │
└─────────────────────┬────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────┐
│           DocumentProcessor (Docling)                     │
│   Docling → Markdown → MarkdownHeaderTextSplitter        │
│   SHA-256 content hashing → 7-day disk cache (.pkl)      │
└─────────────────────┬────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────┐
│         Hybrid Retriever (BM25 + Vector)                  │
│   BM25Retriever (weight 0.4)                             │
│   ChromaDB + IBM Slate 125M embeddings (weight 0.6)      │
│   EnsembleRetriever combines both                        │
└─────────────────────┬────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────┐
│         LangGraph AgentWorkflow (3 Agents)               │
│                                                          │
│  ┌───────────────────┐                                   │
│  │ RelevanceChecker  │ IBM Granite 3.3 8B               │
│  │ CAN_ANSWER /      │ Classifies doc-query match       │
│  │ PARTIAL / NO_MATCH│                                   │
│  └────────┬──────────┘                                   │
│           │ relevant → research                          │
│           │ irrelevant → END                             │
│  ┌────────▼──────────┐                                   │
│  │  ResearchAgent    │ Llama 3.3 70B Instruct           │
│  │  Generates draft  │ Context-grounded answers         │
│  │  answer from docs │                                   │
│  └────────┬──────────┘                                   │
│           │                                              │
│  ┌────────▼──────────┐                                   │
│  │ VerificationAgent │ Checks factual accuracy          │
│  │  Supported: YES/NO│ vs source documents              │
│  └────────┬──────────┘                                   │
│           │ verified → END                               │
│           │ failed → re_research (loop back)            │
└──────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
docchat/
│
├── app.py                          # Gradio UI + session state management
├── agents/
│   ├── workflow.py                 # LangGraph AgentWorkflow (3 agents)
│   ├── research_agent.py           # ResearchAgent — Llama 3.3 70B
│   ├── verification_agent.py       # VerificationAgent — factual check
│   └── relevance_checker.py        # RelevanceChecker — Granite 3.3 8B
├── document_processor/
│   └── file_handler.py             # Docling parser + chunking + caching
├── retriever/
│   └── builder.py                  # Hybrid BM25 + ChromaDB retriever
├── config/
│   ├── settings.py                 # Pydantic settings (env vars)
│   └── constants.py                # File size limits + allowed types
├── utils/
│   └── logging.py                  # Logger setup
├── document_cache/                 # SHA-256 hashed .pkl cache files
├── test/                           # Test PDFs + OCR test cases
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Agent Orchestration | LangGraph (`StateGraph`) |
| Research LLM | Meta Llama 3.3 70B Instruct (IBM Watsonx) |
| Relevance LLM | IBM Granite 3.3 8B Instruct (IBM Watsonx) |
| Embeddings | IBM Slate 125M English Retriever v2 (Watsonx) |
| Document Parser | Docling → Markdown |
| Text Splitter | MarkdownHeaderTextSplitter (H1, H2) |
| Vector Store | ChromaDB |
| Keyword Retriever | BM25Retriever |
| Hybrid Retrieval | EnsembleRetriever (BM25 0.4 + Vector 0.6) |
| UI | Gradio (Citrus theme + custom CSS/JS) |
| Caching | SHA-256 + Pickle (7-day expiry) |
| Config | Pydantic BaseSettings + .env |

---

## 🤖 3 Specialized AI Agents

### 1. RelevanceChecker (IBM Granite 3.3 8B)
```python
# Classifies document-query relevance
response → "CAN_ANSWER" | "PARTIAL" | "NO_MATCH"

# Routing:
CAN_ANSWER / PARTIAL → proceed to ResearchAgent
NO_MATCH → return "Question not related to uploaded documents"
```

### 2. ResearchAgent (Meta Llama 3.3 70B Instruct)
```python
# Generates grounded answers from retrieved context
model_id = "meta-llama/llama-3-3-70b-instruct"
params = {"max_tokens": 300, "temperature": 0.3}

# Context = combined top document chunks
context = "\n\n".join([doc.page_content for doc in documents])
```

### 3. VerificationAgent
```python
# Checks factual accuracy of draft answer vs source documents
# Reports: "Supported: YES/NO" + "Relevant: YES/NO"

# If verification fails → triggers re_research loop
"Supported: NO" or "Relevant: NO" → re_research
else → END
```

---

## 🔄 LangGraph Workflow

```python
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever: EnsembleRetriever

# Node connections
workflow.set_entry_point("check_relevance")
workflow.add_conditional_edges("check_relevance", decide, {
    "relevant": "research",
    "irrelevant": END
})
workflow.add_edge("research", "verify")
workflow.add_conditional_edges("verify", decide_next, {
    "re_research": "research",   # Loop back if verification fails
    "end": END
})
```

---

## 🔍 Hybrid RAG Retrieval

```python
# BM25 — keyword-based retrieval (weight 0.4)
bm25 = BM25Retriever.from_documents(docs)

# ChromaDB — semantic vector retrieval (weight 0.6)
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=WatsonxEmbeddings(model_id="ibm/slate-125m-english-rtrvr-v2")
)

# Ensemble — combines both for best coverage
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25, vector_retriever],
    weights=[0.4, 0.6]
)
```

---

## 📄 Document Processing Pipeline

```python
# Docling parses PDF/DOCX/TXT/MD → Markdown
converter = DocumentConverter()
result = converter.convert(file_path)
markdown_text = result.document.export_to_markdown()

# Split on Markdown headers
splitter = MarkdownHeaderTextSplitter(headers=[("#", "Header 1"), ("##", "Header 2")])
chunks = splitter.split_text(markdown_text)

# SHA-256 content caching (7-day expiry)
file_hash = hashlib.sha256(file_bytes).hexdigest()
cache_path = f"document_cache/{file_hash}.pkl"
```

---

## ⚙️ Configuration

```python
class Settings(BaseSettings):
    CHROMA_DB_PATH: str = "./chroma_db"
    VECTOR_SEARCH_K: int = 10
    HYBRID_RETRIEVER_WEIGHTS: list = [0.4, 0.6]
    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    class Config:
        env_file = ".env"
```

---

## 🚀 How to Run

**Step 1 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 2 — Set environment variables:**
```bash
# .env file
OPENAI_API_KEY=your-key
WATSONX_API_KEY=your-key
```

**Step 3 — Launch DocChat:**
```bash
python app.py
# Opens at http://127.0.0.1:5000
```

**Step 4 — Use the UI:**
- Upload PDF/DOCX/TXT/MD file
- Enter your question
- Click **Submit** → get grounded, verified answer!

---

## 📋 Example Queries (Included)

| Document | Sample Query |
|---|---|
| Google 2024 Environmental Report | "Retrieve the data center PUE efficiency values in Singapore 2nd facility in 2019 and 2022. Also retrieve regional average CFE in Asia Pacific in 2023" |
| DeepSeek-R1 Technical Report | "Summarize DeepSeek-R1 model's performance evaluation on all coding tasks against OpenAI o1-mini model" |

---

## 🎓 Skills Demonstrated

- Multi-agent RAG pipeline with LangGraph StateGraph
- Hybrid retrieval — BM25 + ChromaDB + IBM Watsonx embeddings
- Docling document parsing (PDF, DOCX, TXT, MD → Markdown)
- 3 specialized IBM Watsonx agents — relevance, research, verification
- Meta Llama 3.3 70B + IBM Granite 3.3 8B integration
- IBM Slate 125M embedding model for semantic search
- Conditional re-research loop when verification fails
- SHA-256 content hashing for intelligent document caching
- Session state management in Gradio
- Pydantic BaseSettings for configuration management
- Production-ready architecture with modular design

---

## 📜 Certifications

| Certification | Issuer | Platform |
|---|---|---|
| IBM Data Science Professional Certificate | IBM | Coursera |
| IBM Generative AI Professional Certificate | IBM | Coursera |
| IBM Agentic AI with RAG Certificate | IBM | Coursera |
| IBM RAG and Agentic AI Professional Certificate | IBM | Coursera |

---

## 🤝 Connect with Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Leela%20A-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/leela-a)
[![Gmail](https://img.shields.io/badge/Gmail-attotaleelaissak@gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:attotaleelaissak@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-Leelaissakattaota-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Leelaissakattaota)
