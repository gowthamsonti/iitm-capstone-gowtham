import os
assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY before running this notebook"

from w6_naiverag_practice import (
    documents,
    chunk_documents,
    build_index,
    retrieve,
    ask_rag,
    cost_usd,
    DEFAULT_SYSTEM,
)

print(f"Loaded {len(documents)} documents from Day 1.")
print("Pipeline functions available: chunk_documents, build_index, retrieve, ask_rag, cost_usd")
