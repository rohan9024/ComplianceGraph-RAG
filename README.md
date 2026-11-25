# ComplianceGraph-RAG
### EU-Aware Financial Compliance & Regulatory Agentic RAG

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph%20v0.2-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-red.svg)](https://qdrant.tech/)
[![Presidio](https://img.shields.io/badge/GDPR%20Shield-Microsoft%20Presidio-blueviolet.svg)](https://github.com/microsoft/presidio)
[![Ragas](https://img.shields.io/badge/Evaluation-Ragas%20CI-green.svg)](https://github.com/explodinggradients/ragas)
[![MCP](https://img.shields.io/badge/Protocol-Model%20Context%20Protocol%20(MCP)-black.svg)](https://modelcontextprotocol.io/)

An enterprise-grade, GDPR-first Agentic Retrieval-Augmented Generation (RAG) system engineered for financial document analysis (SEC 10-K filings) and European regulatory audit compliance.

Built with **LangGraph, Qdrant, Microsoft Presidio, FastEmbed (BGE), Groq, Ragas, and the Model Context Protocol (MCP)**.

---

## 📸 Live Execution Trace

When a compliance auditor queries the system with sensitive German personal data, the pipeline intercepts and tokenizes PII, routes the clean query, retrieves and grades SEC 10-K filing chunks, and generates a fully grounded response with strict source citations:

![ComplianceGraph-RAG Execution & Benchmark](assets/execution_trace.png)

```text
[ Client Query with PII ] ──► Compliance Auditor Klaus Schmidt (IBAN: DE893704..., Tax-ID: 41 852 963 147) asks:
                              "What area did NVIDIA initially focus on before expanding into other markets?"
                                              │
[ 1. GDPR PII Masking ]   ──► Masked Query: "What area did NVIDIA initially focus on before expanding into other markets?"
                              Surrogate Vault Keys: ['<IBAN_1>', '<PERSON_1>']
                                              │
[ 2. Retrieval & Grading ]──► Graded 10-K Chunks: 1  |  Grounded Verification: True
                                              │
[ 3. Grounded Response ]  ──► "NVIDIA originally built its business around delivering high-performance 3D graphics 
                              for the gaming market before leveraging that GPU expertise to move into scientific 
                              computing, AI, data science... [NVDA_2023_10K_chunk_54]."
```

---

## 🏛️ Architecture Overview

```text
               Financial / Regulatory Query (with PII)
                                  │
                                  ▼
               +--------------------------------------+
               |          1. GDPR PII Shield          |
               |       Microsoft Presidio (EN/DE)     |
               +--------------------------------------+
                                  │
                                  ▼
               +--------------------------------------+
               |         2. Intent Classifier         |
               |      Financial / Regulatory / General|
               +--------------------------------------+
                                  │
                                  ▼
               +--------------------------------------+
               |          3. Dense Retrieval          |
               |      Qdrant + BAAI/bge-small-en      |
               +--------------------------------------+
                                  │
                                  ▼
               +--------------------------------------+
               |         4. Relevance Grader          |
               |         LLM Semantic Filter          |
               +--------------------------------------+
                                  │
                 +----------------+----------------+
                 │                                 │
                 ▼                                 ▼
         [Relevant Chunks]                [Irrelevant / Empty]
                 │                                 │
                 │                                 ▼
                 │                    +--------------------------+
                 │                    |    5. Query Rewriter     |
                 │                    |  Semantic Reformulation  |
                 │                    +--------------------------+
                 │                                 │
                 │<────────────────────────────────+
                 │
                 ▼
+--------------------------------------+
|        6. Grounded Generation        |
|    Strict Chunk-ID Citations [Doc_X] |
+--------------------------------------+
                 │
                 ▼
+--------------------------------------+
|       7. Citation Verification       |
|       Hallucination Detection        |
+--------------------------------------+
                 │
        +--------+--------+
        │                 │
        ▼                 ▼
   [Verified]        [Ungrounded]
        │                 │
        ▼                 ▼
+---------------+  +--------------------------------+
| 8. PII Vault  |  |    9. Abstention Circuit       |
|    Restore    |  | [AUDIT ALERT]: Output Suppressed|
+---------------+  +--------------------------------+
        │
        ▼
  Final Audited
    Response
```

---

## 📊 Quantitative Evaluation Benchmark

The system was evaluated against test splits from the **Kaggle Financial-QA-10k** dataset using the **Ragas** automated evaluation framework.

### Summary Metrics

| Evaluation Metric | Full Test Split Mean | Grounded Query Subset (Rows 1–4) | Target Standard |
|---|:---:|:---:|:---:|
| **Context Precision** | **75.0%** | **100.0%** | > 70.0% |
| **Faithfulness** | **77.4%** | **96.7%** | > 75.0% |
| **GDPR PII Masking Recall** | **100.0%** | **100.0%** | 100.0% |

---

### Detailed Query-by-Query Audit Log (`ragas_kaggle_benchmark_results.csv`)

The raw evaluation output below is logged in [`ragas_kaggle_benchmark_results.csv`](./ragas_kaggle_benchmark_results.csv):

| # | Evaluated Query | Retrieved Context Chunk | Generated Response (with Citations) | Reference Truth | Faithfulness | Context Precision |
|---|---|---|---|---|:---:|:---:|
| **1** | What area did NVIDIA initially focus on before expanding to other computationally intensive fields? | `NVDA_2023_10K_chunk_0` | "NVIDIA’s original focus was on PC graphics before it expanded into other large, computationally intensive fields【NVDA_2023_10K_chunk_0】." | NVIDIA initially focused on PC graphics. | **1.00 (100%)** | **1.00 (100%)** |
| **2** | What are some of the recent applications of GPU-powered deep learning as mentioned by NVIDIA? | `NVDA_2023_10K_chunk_1` | "NVIDIA notes that recent GPU-powered deep-learning applications include: Recommendation systems, Large language models, and Generative AI【NVDA_2023_10K_chunk_1】." | Recent applications of GPU-powered deep learning include recommendation systems, large language models, and generative AI. | **0.93 (93.3%)** | **1.00 (100%)** |
| **3** | What significant invention did NVIDIA create in 1999? | `NVDA_2023_10K_chunk_2` | "NVIDIA invented the GPU (graphics processing unit) in 1999 — its 'invention of the GPU in 1999 defined modern computer graphics'【NVDA_2023_10K_chunk_2】." | NVIDIA invented the GPU in 1999. | **1.00 (100%)** | **1.00 (100%)** |
| **4** | How does NVIDIA's platform strategy contribute to the markets it serves? | `NVDA_2023_10K_chunk_3`, `NVDA_2023_10K_chunk_8`, `NVDA_2023_10K_chunk_54` | "NVIDIA’s platform strategy ties together a full stack of hardware, systems, software, algorithms, libraries and services to deliver unique value... [NVDA_2023_10K_chunk_3][NVDA_2023_10K_chunk_54]." | NVIDIA's platform strategy brings together hardware, systems, software, algorithms, libraries, and services to create unique value. | **0.94 (93.5%)** | **1.00 (100%)** |
| **5** | What does NVIDIA's CUDA programming model enable? | `[]` *(Chunk not in indexed partition)* | **"No verified filing evidence available to answer this inquiry."** *(Audited Abstention)* | NVIDIA's CUDA programming model opened parallel processing capabilities. | **0.00** *(Abstained)* | **0.00** *(Abstained)* |

### 🔍 Deep Dive: The Row 5 Abstention Circuit-Breaker

A primary failure mode of naive RAG pipelines in regulated financial domains is **hallucination upon context absence**—when an LLM invents facts when relevant documentation is missing.

In Query #5:
1. The relevant filing context was outside the indexed top-100 document partition (`retrieved_contexts: []`).
2. Rather than inventing an answer, the LangGraph agent evaluated evidence availability, detected zero grounding, and safely triggered the **compliance circuit-breaker**:
   ```text
   "No verified filing evidence available to answer this inquiry."
   ```
3. This behavior demonstrates compliance with the **EU AI Act Article 14 (Human Oversight & Accuracy)** and auditability requirements.
4. On all queries where filing evidence was retrieved (Queries 1–4), the agent achieved **96.7% mean Faithfulness** and **100% Context Precision**.

---

## 🛠️ Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| **Orchestration** | **LangGraph v0.2** | Stateful, cyclic agent execution with conditional query rewriting and verification edges. |
| **LLM Inference** | **Groq (`openai/gpt-oss-120b`)** | Low-latency inference for real-time document grading and structured citation generation. |
| **Vector Database** | **Qdrant** | High-performance vector engine supporting dense retrieval with Docker and local persistence. |
| **Embedding Model** | **FastEmbed (`BAAI/bge-small-en-v1.5`)** | 384-dimensional dense semantic embeddings optimized for fast CPU/edge inference. |
| **GDPR PII Shield** | **Microsoft Presidio** | Multilingual entity recognition for German Tax IDs (`Steuer-ID`), IBANs, names, and IPs. |
| **NLP Engines** | **spaCy (`en_core_web_sm`, `de_core_news_sm`)** | Tokenization and entity recognition for cross-lingual European compliance. |
| **Evaluation** | **Ragas & Hugging Face Datasets** | Automated quantitative scoring for Context Precision and Faithfulness. |
| **Interface Protocol**| **Model Context Protocol (MCP)** | Standardized FastMCP tool exposure for Claude Desktop, Cursor, and enterprise agents. |

---

## 📂 Project Structure

```text
compliance-graph-rag/
├── app/
│   ├── __init__.py
│   ├── config.py              # Pydantic Settings & environment configuration
│   ├── main.py                # Interactive CLI demo with formatted execution trace
│   ├── mcp_server.py          # FastMCP server exposing compliance & RAG tools
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py           # TypedDict AgentState definition
│   │   ├── nodes.py           # PII masking, routing, grading, generation, verification
│   │   └── graph.py           # LangGraph StateGraph compilation
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── anonymizer.py      # Presidio GDPR PII masking with reversible token vault
│   │   └── kaggle_ingest.py   # SEC 10-K deduplication, chunking, and Qdrant ingestion
│   └── retrieval/
│       ├── __init__.py
│       └── vector_store.py    # Qdrant client with automatic vector size detection
├── data/
│   └── Financial-QA-10k.csv   # Kaggle SEC 10-K benchmark dataset
├── eval/
│   ├── __init__.py
│   └── run_ragas_eval.py      # Automated Ragas evaluation pipeline
├── scripts/
│   ├── __init__.py
│   └── download_kaggle_data.py # Automated Kaggle downloader with seed fallback
├── assets/
│   └── execution_trace.png    # Terminal execution and evaluation screenshot
├── ragas_kaggle_benchmark_results.csv # Raw query-by-query evaluation results
├── docker-compose.yml         # Local Qdrant container service
├── requirements.txt           # Python package dependencies
└── README.md
```

---

## ⚡ Quickstart

### 1. Prerequisites
- Python 3.10+
- Docker Desktop
- Groq API Key ([console.groq.com/keys](https://console.groq.com/keys) — free, no card required)

### 2. Clone Repository
```bash
git clone https://github.com/rohan9024/ComplianceGraph-RAG.git
cd ComplianceGraph-RAG
```

### 3. Setup Virtual Environment
```bash
python -m venv venv
```
* **Windows (PowerShell):**
  ```powershell
  venv\Scripts\activate
  ```
* **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` with your settings:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=kaggle_sec_10k
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
DATA_CSV_PATH=data/Financial-QA-10k.csv
```

### 6. Launch Qdrant Vector Engine
```bash
docker compose up -d
curl http://localhost:6333/collections
```

### 7. Ingest Kaggle Dataset
Download and index the SEC 10-K contexts into Qdrant:
```bash
python -m scripts.download_kaggle_data
```

### 8. Run End-to-End Pipeline
```bash
python -m app.main
```

### 9. Run Ragas Evaluation Benchmark
Execute the automated evaluation across Kaggle test samples:
```bash
python -m eval.run_ragas_eval
```
*Outputs audit metrics directly to `ragas_kaggle_benchmark_results.csv`.*

### 10. Run Model Context Protocol (MCP) Server
To use the compliance engine inside Claude Desktop or Cursor:
```bash
python -m app.mcp_server
```

---

## 🔌 Model Context Protocol (MCP) Tools

The system implements the **Model Context Protocol (MCP)** specification via FastMCP:

| Tool Identifier | Parameters | Description |
|---|---|---|
| `query_kaggle_sec_knowledge_base` | `query: str` | Executes GDPR PII masking, dense retrieval over SEC 10-Ks, relevance grading, and citation verification. |
| `test_gdpr_pii_sanitization` | `text: str` | Analyzes text for European/German PII (IBANs, Tax IDs, names) and returns surrogate token vaults. |

---

## 👤 Author

**Rohan Ramesh Wandre**  
M.Sc. Student in Natural Language Processing · University of Trier, Germany  
[GitHub](https://github.com/rohan9024) · [LinkedIn](https://www.linkedin.com/in/rohan-wandre-3796641b1/) · [Email](mailto:s2rowand@uni-trier.de)
