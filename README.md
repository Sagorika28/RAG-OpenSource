# RAG-OS — Modular RAG Pipeline

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-opensource.streamlit.app/)

Production-ready fully open-source, self-hostable Retrieval-Augmented Generation (RAG) pipeline. Designed for ingesting technical PDFs (manuals, standards, newsletters, booklets, flyers, slides) and answering questions with cited evidence.

## Features

- **Modular architecture**: Swap parsers, embeddings, rerankers, and LLMs via YAML config
- **Adaptive chunking**: Section-aware, block, page-level strategies based on document type
- **Qdrant vector store**: Local embedded mode (no Docker) or server mode
- **Retrieval + optional reranking + optional LLM generation**
- **Evaluation harness**: Recall@k, MRR@k, nDCG@k with latency profiling
- **Streamlit demo UI**: Question → evidence → answer with citations
- **CPU-first**: Runs on MacBook Air (Apple Silicon) without GPU

## Quick Start

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Your PDFs

Place your PDF files into the `data/pdfs/` directory:

```bash
mkdir -p data/pdfs
# Copy your PDFs here
cp /path/to/your/*.pdf data/pdfs/
```

### 3. Ingest PDFs

```bash
# Ingest using CPU config (default)
python -m src.pipeline.ingest --config configs/cpu.yaml

# Or recreate the index from scratch
python -m src.pipeline.ingest --config configs/cpu.yaml --recreate
```

### 4. Run a Query (CLI)

```bash
# Query with LLM generation (requires Ollama running)
python -m src.pipeline.query "What are the safety precautions for handling refrigerants?" --config configs/cpu.yaml

# Query without LLM (retrieval only)
python -m src.pipeline.query "What are the safety precautions?" --config configs/cpu.yaml --no-generate
```

### 5. Launch Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 6. Run Evaluation

First, edit `data/eval/questions.jsonl` to match your actual PDF filenames and questions:

```bash
# Run eval and generate report
python -m src.eval.run_eval --config configs/cpu.yaml --output outputs/eval_report.json

# Compare configs
python -m src.eval.run_eval --config configs/gpu.yaml --output outputs/eval_report_gpu.json
```

### 7. (Optional) Start Qdrant via Docker

For production mode with persistent Qdrant server:

```bash
docker compose up -d
```

Then switch your config to `mode: server`:

```yaml
# In configs/cpu.yaml or configs/gpu.yaml
qdrant:
  mode: server
  url: http://localhost:6333
```

### 8. (Optional) Ollama Setup

To use the LLM generation feature, install and run [Ollama](https://ollama.ai):

```bash
# Install Ollama (macOS)
brew install ollama

# Pull the default model
ollama pull qwen2.5:3b

# Ollama runs automatically as a service, or start manually:
ollama serve
```

## Project Structure

```
RAG-OS/
├── configs/
│   ├── cpu.yaml              # CPU-only config (default)
│   └── gpu.yaml              # GPU-accelerated config
├── data/
│   ├── pdfs/                 # Place PDFs here
│   └── eval/
│       └── questions.jsonl   # Evaluation dataset template
├── src/
│   ├── core/                 # Shared types, config, utils
│   ├── parsing/              # PDF parsers (Docling + PyMuPDF)
│   ├── chunking/             # Adaptive structural chunking
│   ├── metadata/             # Document type classification
│   ├── embeddings/           # BGE-small (CPU) / BGE-M3 (GPU)
│   ├── index/                # Qdrant vector store
│   ├── rerank/               # BGE cross-encoder reranker
│   ├── generation/           # Ollama LLM generation
│   ├── pipeline/             # Ingest + query orchestration
│   └── eval/                 # Evaluation harness
├── app/
│   └── streamlit_app.py      # Demo UI
├── docker-compose.yml        # Optional Qdrant server
├── requirements.txt
└── README.md
```

## Configuration

The system is config-driven. Switch between CPU and GPU profiles:

| Feature | `cpu.yaml` | `gpu.yaml` |
|---------|-----------|-----------|
| Embeddings | BGE-small-en-v1.5 (384d) | BGE-M3 (1024d) |
| Reranker | Off (speed) | BGE-reranker-large |
| Qdrant | Local embedded | Server (Docker) |
| Device | CPU | CUDA |

## ☁️ Cloud Deployment (Streamlit Cloud)

To deploy this app for free on **Streamlit Cloud** using Groq for fast inference:

1.  **Fork/Push this repo to GitHub.**
2.  **Get a Groq API Key** from [console.groq.com](https://console.groq.com).
3.  **Deploy on Streamlit Cloud**:
    - Connect your GitHub repo.
    - In "Advanced settings" -> "Secrets", add:
      ```toml
      GROQ_API_KEY = "gsk_..."
      ```
4.  **Configuration**:
    - The app defaults to `groq.yaml`.
    - This config uses **Groq Cloud API** for generation (Instant speed).
    - Embeddings run on CPU (Cloud) or MPS (Mac).

### Local Setup with Groq
To run locally with Groq (instead of Ollama):
1.  Add your key to `.env`:
    ```bash
    GROQ_API_KEY=gsk_...
    ```
2.  Run with Groq config:
    ```bash
    streamlit run app/streamlit_app.py
    # Select 'groq.yaml' in the sidebar
    ```

## Evaluation

The eval harness computes:

- **Retrieval metrics**: Recall@k, MRR@k, nDCG@k (k=1,3,5,10)
- **Latency metrics**: embed, search, rerank, generate (mean, p50, p95)
- **Index stats**: document/chunk counts, vector dimensions

Edit `data/eval/questions.jsonl` with your test questions and expected source documents. Each line is a JSON object:

```json
{"question": "Your question here", "gold_sources": ["expected_file.pdf"], "metadata": {"topic": "safety"}}
```

## License

Open source — see LICENSE file for details.
