
# ComplianceGraph-RAG
### EU-Aware Financial Compliance & Regulatory Agentic RAG

An agentic Retrieval-Augmented Generation (RAG) system designed for financial document analysis, GDPR-oriented PII protection, and evidence-grounded question answering.

Built using **Python, LangGraph, Qdrant, Microsoft Presidio, FastEmbed, LangChain, Ragas, and Model Context Protocol (MCP)**.

The system processes financial filing queries through a self-corrective agent pipeline with PII masking, semantic retrieval, relevance grading, query rewriting, citation verification, and controlled response generation.

---

## Architecture Overview

```text
          Financial / Regulatory Query
                       |
                       v
          +-------------------------+
          |    1. GDPR PII Shield   |
          |   Microsoft Presidio   |
          +-------------------------+
                       |
                       v
          +-------------------------+
          |   2. Intent Classifier  |
          | Financial / Regulatory |
          |        / General       |
          +-------------------------+
                       |
                       v
          +-------------------------+
          |   3. Dense Retrieval    |
          | Qdrant + FastEmbed      |
          | multilingual-e5-large   |
          +-------------------------+
                       |
                       v
          +-------------------------+
          |   4. Relevance Grader   |
          |   LLM-based Filtering   |
          +-------------------------+
                       |
              +--------+--------+
              |                 |
              v                 v
         Relevant           Irrelevant
          Chunks              Chunks
              |                 |
              |                 v
              |        +-------------------+
              |        | 5. Query Rewriter |
              |        | Adaptive Retrieval|
              |        +-------------------+
              |                 |
              |<----------------+
              |
              v
          +-------------------------+
          | 6. Grounded Generation  |
          |   Chunk-ID Citations   |
          +-------------------------+
                       |
                       v
          +-------------------------+
          | 7. Citation Verification|
          | Hallucination Detection |
          +-------------------------+
                       |
              +--------+--------+
              |                 |
              v                 v
           Verified         Ungrounded
              |                 |
              v                 v
          +----------+    +-------------+
          | 8. PII   |    |   Abstain   |
          | Restore  |    | Audit Alert |
          +----------+    +-------------+
              |
              v
        Final Response
```

---

## Key Features

- **GDPR-Oriented PII Protection:** Integrates Microsoft Presidio to detect and mask sensitive information, including names, email addresses, phone numbers, German tax IDs, IBANs, and IP addresses.

- **Reversible PII Token Vault:** Replaces detected entities with surrogate tokens such as `<IBAN_1>` and `<PERSON_1>`, maintaining a mapping for controlled restoration in the final response.

- **Self-Corrective Agentic RAG:** Uses LangGraph to orchestrate query classification, document retrieval, relevance grading, adaptive query rewriting, and response verification.

- **Semantic Vector Search:** Uses Qdrant and FastEmbed with `intfloat/multilingual-e5-large` for 1024-dimensional multilingual dense embeddings.

- **Citation-Grounded Generation:** Requires generated financial answers to reference retrieved filing chunks using their unique identifiers.

- **Hallucination Verification:** Implements an LLM-based verification stage to check whether generated claims are supported by retrieved evidence.

- **Audit Abstention Mechanism:** Suppresses responses when citation verification identifies unsupported claims.

- **Model Context Protocol (MCP):** Exposes financial knowledge-base querying and PII sanitization through standardized MCP tools.

- **Quantitative Evaluation:** Uses Ragas to evaluate retrieval and generation quality against the Kaggle Financial-QA-10k benchmark.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Agent Orchestration | LangGraph |
| LLM Integration | LangChain |
| Vector Database | Qdrant |
| Embedding Framework | FastEmbed |
| Embedding Model | intfloat/multilingual-e5-large |
| PII Detection | Microsoft Presidio |
| NLP Pipelines | spaCy (English & German) |
| Benchmark Dataset | Kaggle Financial-QA-10k |
| Evaluation | Ragas |
| Dataset Processing | Pandas, Hugging Face Datasets |
| API / Protocol | Model Context Protocol (MCP) |
| Infrastructure | Docker Compose |

---

## Quantitative Evaluation

### Kaggle Financial-QA-10k Benchmark

Evaluated using the Ragas framework against financial question-answering examples from the Kaggle Financial-QA-10k dataset.

| Evaluation Metric | Result |
|---|---:|
| Context Precision | **0.80 (80%)** |
| Faithfulness | See evaluation CSV |
| Answer Relevancy | See evaluation CSV |

**Context Precision: 80%**

The measured result indicates that the retrieval and relevance-filtering pipeline achieved 0.80 Context Precision on the evaluated sample.

The complete evaluation output is saved in:

`ragas_kaggle_benchmark_results.csv`

---

## Dataset

**Kaggle: Financial-QA-10k**

Dataset: [yousefsaeedian/financial-qa-10k](https://www.kaggle.com/datasets/yousefsaeedian/financial-qa-10k)

Contains financial question-answering examples with associated SEC 10-K filing contexts and metadata.

Key fields:

- `question` – Financial question.
- `answer` – Ground-truth answer.
- `context` – Supporting filing excerpt.
- `ticker` – Company stock ticker.
- `filing` – Filing year and type.

The ingestion pipeline deduplicates filing contexts and indexes them into Qdrant.

---

## Project Structure

```text
compliance-graph-rag/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── mcp_server.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── anonymizer.py
│   │   └── kaggle_ingest.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── vector_store.py
│   │
│   └── agent/
│       ├── __init__.py
│       ├── state.py
│       ├── nodes.py
│       └── graph.py
│
├── data/
├── scripts/
│   ├── __init__.py
│   └── download_kaggle_data.py
│
├── eval/
│   ├── __init__.py
│   └── run_ragas_eval.py
│
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quickstart

### 1. Prerequisites

- Python 3.11
- Docker Desktop
- Kaggle account
- OpenAI API key

### 2. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/ComplianceGraph-RAG.git

cd ComplianceGraph-RAG
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```powershell
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip setuptools wheel

pip install -r requirements.txt
```

### 5. Install spaCy Language Models

```bash
python -m spacy download en_core_web_sm

python -m spacy download de_core_news_sm
```

### 6. Configure Environment Variables

Windows:

```powershell
Copy-Item .env.example .env
```

Linux / macOS:

```bash
cp .env.example .env
```

Configure `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=kaggle_sec_10k

EMBEDDING_MODEL=intfloat/multilingual-e5-large

DATA_CSV_PATH=data/Financial-QA-10k.csv
```

### 7. Launch Qdrant

```bash
docker compose up -d
```

Verify the database:

```bash
curl http://localhost:6333/collections
```

### 8. Download Kaggle Dataset

Configure Kaggle API credentials, then execute:

```bash
python -m scripts.download_kaggle_data
```

Alternatively, download the dataset manually from Kaggle and place `Financial-QA-10k.csv` inside the `data/` directory.

### 9. Run the Application

```bash
python -m app.main
```

This initializes the ingestion pipeline, embeds financial filing contexts, indexes them in Qdrant, and executes the agent against a sample financial query.

### 10. Run Ragas Evaluation

```bash
python -m eval.run_ragas_eval
```

The evaluation generates:

```text
ragas_kaggle_benchmark_results.csv
```

### 11. Run MCP Server

```bash
python -m app.mcp_server
```

---

## MCP Tools

The application exposes two MCP tools:

| Tool | Description |
|---|---|
| `query_kaggle_sec_knowledge_base` | Queries the financial filing knowledge base through the LangGraph pipeline. |
| `test_gdpr_pii_sanitization` | Detects and masks German and European PII. |

These tools provide an integration point for MCP-compatible AI clients and agent workflows.

---

## Future Improvements

- Implement hybrid retrieval combining dense embeddings and BM25.
- Introduce cross-encoder reranking for improved retrieval precision.
- Add persistent and access-controlled PII token vault storage.
- Extend document ingestion to additional European regulatory sources.
- Introduce automated regression testing for retrieval and generation.
- Add observability and tracing for agent execution.
- Improve multilingual retrieval and German financial document support.

---

## Author

**Rohan Ramesh Wandre**

M.Sc. Natural Language Processing | University of Trier

Interested in NLP, Agentic AI, Retrieval-Augmented Generation, multilingual systems, and privacy-aware AI infrastructure.

---

*Built as an applied research and engineering project exploring reliable, evidence-grounded financial AI systems.*
