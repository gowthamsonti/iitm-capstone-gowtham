# MP2 Mini-RAG: Sherlock Holmes Question Answering System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline over a small corpus of Sherlock Holmes stories.

The system:

1. Loads story summaries from text files.
2. Splits documents into chunks.
3. Generates embeddings using OpenAI's `text-embedding-3-small`.
4. Stores embeddings in a Qdrant vector database.
5. Retrieves relevant chunks for a user query.
6. Uses `gpt-4o-mini` to generate answers with citations.

---

## Corpus

The corpus contains five Sherlock Holmes stories:

- 01_red_headed_league.txt
- 02_speckled_band.txt
- 03_blue_carbuncle.txt
- 04_engineers_thumb.txt
- 05_scandal_in_bohemia.txt

---

## Project Structure

```text
learner/
│
├── corpus/
│   ├── 01_red_headed_league.txt
│   ├── 02_speckled_band.txt
│   ├── 03_blue_carbuncle.txt
│   ├── 04_engineers_thumb.txt
│   └── 05_scandal_in_bohemia.txt
│
├── data/
│   ├── predefined_questions.jsonl
│   └── learner_questions.jsonl
│
├── mp2_rag.py
├── mp2_reflection.md
├── mp2_validation.txt
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.10+
- OpenAI API Key
- Qdrant Cloud Account

---

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages include:

```bash
openai
qdrant-client
python-dotenv
```

---

## Running the Project

### Step 1: Build the Vector Database

```bash
python mp2_rag.py ingest
```

Expected output:

```text
→ Loading corpus…
→ Chunking…
→ Setting up Qdrant collection…
→ Ingesting…

✓ Done. Try: python mp2_rag.py ask
```

This command:

- Loads all documents
- Creates chunks
- Generates embeddings
- Stores vectors in Qdrant

---

### Step 2: Ask Questions

```bash
python mp2_rag.py ask
```

Example:

```text
? Who was John Clay?
```

Example response:

```text
John Clay was a criminal and master thief involved in a bank robbery scheme.

Sources:
    - The Red-Headed League — Chunk 11
    - The Red-Headed League — Chunk 9
```

---

### Step 3: Run Validation

```bash
python mp2_rag.py validate
```

The validator executes both:

- predefined_questions.jsonl
- learner_questions.jsonl

Sample output:

```text
✓ learner_1
✓ learner_2
✓ learner_3
...
```

---

## Retrieval Pipeline

```text
Documents
    ↓
Chunking
    ↓
OpenAI Embeddings
    ↓
Qdrant Vector Store
    ↓
Similarity Search
    ↓
Retrieved Context
    ↓
GPT-4o-mini
    ↓
Answer + Citations
```

---

## Chunking Strategy

The implementation uses fixed-size character chunks.

Parameters:

```python
TARGET_CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
```

Each chunk maintains metadata:

- source
- title
- section
- text

---

## Models Used

### Embedding Model

```text
text-embedding-3-small
```

Dimension:

```text
1536
```

### LLM

```text
gpt-4o-mini
```

Used for answer generation.

---

## Validation Results

### Learner Question Validation

| Metric | Result |
|----------|----------|
| Questions | 10 |
| Source Matches | 10 |
| Source Accuracy | 100% |

Result:

```text
Source-match: 10/10
```

All learner-authored questions retrieved the correct source document.

---

## Interpretation of Results

### Source Match

Measures whether the retrieved citations contain the expected document.

Example:

```text
Expected:
01_red_headed_league.txt

Retrieved:
01_red_headed_league.txt

Result:
PASS
```

A perfect source-match score indicates successful retrieval.

### Fact Match

Measures whether expected keywords appear in the generated answer.

Fact-match failures may occur even when retrieval is correct because the language model uses paraphrases or synonyms.

### Latency

Reported in milliseconds.

Includes:

- Query embedding
- Vector retrieval
- LLM generation

Typical latency observed:

```text
2–5 seconds
```

---

## Improvements for Future Work

Potential enhancements:

- Paragraph-aware chunking
- Hybrid retrieval
- BM25 + Vector Search
- Reranking
- Metadata filtering
- Retrieval evaluation metrics

---

## Reflection Summary

Key achievements:

- Successfully built an end-to-end RAG pipeline.
- Integrated OpenAI and Qdrant Cloud.
- Achieved 100% source retrieval accuracy on the learner validation set.
- Generated source-cited answers using GPT-4o-mini.

---

## Author

Gowtham S

MP2 – Mini-RAG Project