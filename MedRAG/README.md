# 🩺 Medical PDF RAG Assistant

A modular Retrieval-Augmented Generation (RAG) system that lets patients and clinicians upload a medical PDF report, receive a plain-language summary, and ask natural-language questions about it.

---

## Project Structure

```
medical_rag_app/
├── app.py            ← Gradio UI (entry point)
├── llm.py            ← LLM loading (medgemma-4b-it + MediLlama-3.2)
├── pdf_loader.py     ← PDF parsing & text chunking
├── vector_store.py   ← Embeddings, FAISS index, guardrail classifier
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/medical_rag_app.git
cd medical_rag_app
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** The app requires a CUDA-capable GPU with ≥16 GB VRAM (both LLMs are loaded simultaneously in 4-bit NF4 quantisation).  
> If you only have a CPU or small GPU, load one model at a time and adjust `device_map` / `device` accordingly.

### 4. Authenticate with Hugging Face

`google/medgemma-4b-it` require you to accept their licences on the Hub and provide a token:

```bash
huggingface-cli login
```

Or set the environment variable:

```bash
export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

---

## Running the App

```bash
python app.py
```

Gradio will print a local URL (e.g. `http://127.0.0.1:7860`).  
Open it in your browser.

---

## Tabs

| Tab | What it does |
|-----|-------------|
| 📄 Upload PDF | Parse, chunk (800 chars / 50 overlap) and FAISS-index the report |
| 📝 Summarise PDF | Stream a progressive summary of the first 15 chunks via MediLlama |
| 💬 Chat with PDF | RAG-powered Q&A with medgemma; optional guardrail blocks off-topic queries |

---

## Guardrail (Optional)

The chat tab supports a `LogisticRegression` classifier that filters out non-medical or potentially harmful queries before they reach the LLM.

To enable it:

1. Generate positive / negative query sets (see the original notebook).
2. Embed them with `vector_store.embed_queries(...)`.
3. Train with `vector_store.train_guardrail(good_embeds, poor_embeds)`.
4. Assign the returned classifier to `guardrail_clf` in `app.py`.

---

## Module Reference

### `llm.py`
| Function | Returns |
|----------|---------|
| `load_chat_llm()` | `(tokenizer, model, pipeline)` for medgemma-4b-it |
| `load_summ_llm()` | `(tokenizer, model, pipeline)` for MediLlama-3.2 |

### `pdf_loader.py`
| Function | Returns |
|----------|---------|
| `load_pdf(path)` | Raw `list[Document]` |
| `split_documents(docs, ...)` | Chunked `list[Document]` |
| `load_and_split(path, ...)` | Convenience: load + split |

### `vector_store.py`
| Function | Returns |
|----------|---------|
| `load_embedding_model()` | `HuggingFaceEmbeddings` |
| `build_vector_store(docs, emb)` | `FAISS` |
| `get_retriever(vs, k)` | LangChain retriever |
| `train_guardrail(good, poor)` | `LogisticRegression` |
| `is_medical_query(q, emb, clf)` | `bool` |

---

## Requirements

- Python ≥ 3.10
- CUDA-capable GPU (recommended ≥ 16 GB VRAM)
- Hugging Face account with access to gated models
