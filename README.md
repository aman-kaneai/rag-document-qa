# 📄 RAG Document Q&A System

> **Retrieval-Augmented Generation pipeline** for intelligent, source-grounded document question answering.

![CI](https://github.com/amanmishra/rag-document-qa/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** grounds LLM answers in your own documents:

```
Your Documents → Chunked → Embedded → FAISS Index
                                           ↓
User Question → Embedded → Top-K Retrieval → LLM → Grounded Answer
```

This eliminates hallucinations by ensuring the model answers only from retrieved evidence.

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| 🔍 **Smart chunking** | Sliding-window text splitting with configurable overlap |
| ⚡ **FAISS vector store** | Cosine-similarity retrieval over millions of chunks |
| 🤖 **LLM integration** | OpenAI-compatible chat completions (GPT-4o-mini default) |
| 📊 **Built-in evaluation** | Faithfulness, relevance, precision, recall metrics |
| 🧪 **21 unit tests** | Full pytest coverage across pipeline & evaluator |
| 🖥️ **CLI + Demo mode** | Works without API keys for local exploration |
| 🔄 **CI/CD** | GitHub Actions across Python 3.10, 3.11, 3.12 |

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/amanmishra/rag-document-qa.git
cd rag-document-qa
pip install -r requirements.txt
```

### 2. Set your OpenAI key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Ingest documents

```bash
python main.py ingest --files report.pdf notes.txt research.md
```

### 4. Query

```bash
python main.py query --question "What are the key findings?"
```

### 5. Interactive chat

```bash
python main.py chat
```

---

## 🎮 Demo Mode (no API key needed)

Run without installing heavy dependencies to explore the CLI:

```bash
python main.py query --question "What is RAG?"
python main.py chat
```

Demo mode uses a small built-in knowledge base about AI/ML concepts.

---

## 🏗️ Architecture

```
rag-document-qa/
├── src/
│   ├── rag_pipeline.py    # Core: TextProcessor, VectorStore, LLMClient, RAGPipeline
│   └── evaluator.py       # RAGEvaluator with faithfulness/relevance/precision/recall
├── tests/
│   └── test_rag.py        # 21 unit tests
├── .github/
│   └── workflows/ci.yml   # GitHub Actions CI
├── main.py                # CLI entrypoint
└── requirements.txt
```

### Key components

**`TextProcessor`** — Loads PDF / TXT / MD files and splits them into overlapping chunks using a sliding window.

**`VectorStore`** — Encodes chunks with `sentence-transformers` and indexes them in FAISS for sub-millisecond cosine-similarity retrieval.

**`LLMClient`** — Builds a context-aware prompt from retrieved chunks and calls an OpenAI-compatible chat endpoint.

**`RAGPipeline`** — Orchestrates ingestion, retrieval, and generation with query caching and index persistence.

**`RAGEvaluator`** — Computes faithfulness, answer relevance, context precision, and context recall without external dependencies.

---

## ⚙️ Configuration

All parameters are CLI flags — no config files needed:

```bash
python main.py ingest \
  --files doc.pdf \
  --chunk-size 512 \
  --chunk-overlap 64 \
  --embedding-model all-MiniLM-L6-v2 \
  --index-dir my_index

python main.py query \
  --question "Summarise the methodology" \
  --top-k 5 \
  --llm-model gpt-4o \
  --output-format json
```

---

## 📊 Evaluation

```bash
python src/evaluator.py   # quick self-test
```

Metrics are weighted into an **Overall Score**:

| Metric | Weight | Meaning |
|--------|--------|---------|
| Faithfulness | 35% | Answer grounded in retrieved context |
| Answer Relevance | 30% | Answer addresses the question |
| Context Precision | 20% | Retrieved chunks are on-topic |
| Context Recall | 15% | Relevant information was retrieved |

---

## 🧪 Tests

```bash
pip install pytest
pytest tests/ -v
```

21 tests covering chunking, embeddings mock, response serialisation, and all evaluator metrics.

---

## 🗺️ Roadmap

- [ ] Streamlit web UI
- [ ] Multi-modal support (images in PDFs)
- [ ] Re-ranking with cross-encoders
- [ ] RAGAS / TruLens integration
- [ ] LangChain / LlamaIndex adapters
- [ ] Docker + FastAPI deployment

---

## 👤 Author

**Aman Mishra** — AI/ML Engineer & Software Developer  
📧 amanmishrabmu@gmail.com  
📞 +91 7007517228  
🔗 [LinkedIn](https://linkedin.com/in/jaya-mishra750881283)

---

## 📄 License

MIT — free to use, modify, and distribute.
