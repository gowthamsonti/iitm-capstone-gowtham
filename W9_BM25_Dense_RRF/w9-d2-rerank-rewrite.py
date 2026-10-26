import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# ── Cross-encoder check — HARD ASSERTION ──
try:
    from sentence_transformers import CrossEncoder
except ImportError as e:
    raise ImportError(
        "\n\n"
        "═══════════════════════════════════════════════════════════════\n"
        "SENTENCE-TRANSFORMERS NOT INSTALLED\n"
        "═══════════════════════════════════════════════════════════════\n"
        "This notebook requires sentence-transformers (yesterday's homework).\n\n"
        "Run:\n"
        "   pip install sentence-transformers\n\n"
        "Then reload the kernel and re-run this cell.\n"
        "═══════════════════════════════════════════════════════════════\n"
    ) from e

# Try loading — this will use the cache if the download already happened
print("Loading cross-encoder (from cache if pre-downloaded)...")
try:
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # Verify it works with a quick scoring call
    _ = reranker.predict([('test query', 'test document')])
    print("✓ Cross-encoder loaded and functional.")
except Exception as e:
    raise RuntimeError(
        "\n\nCross-encoder model failed to load. Likely causes:\n"
        "  1. Model wasn't downloaded yesterday (do the Day 1 homework)\n"
        "  2. HuggingFace access blocked by your network — talk to your instructor\n"
        f"\nOriginal error: {e}"
    ) from e

# Load key-value pairs from the .env file into os.environ
load_dotenv(override=True)
# ── Standard env + corpus setup ──
assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY"
assert os.environ.get("QDRANT_URL"),     "Set QDRANT_URL"
assert os.environ.get("QDRANT_API_KEY"), "Set QDRANT_API_KEY"

sys.path.insert(0, str(Path.cwd()))
from wk09_pipeline import (
    load_acme_corpus, load_test_queries,
    simple_tokenize, build_bm25_index, bm25_search,
    rrf_fuse,
)

corpus  = load_acme_corpus()
queries = load_test_queries()
print(f"✓ Loaded {len(corpus)} docs and {len(queries)} test queries.")

#4 Reranker in code
def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """Score each (query, candidate.text) pair; return top_k by rerank_score.
    
    candidates: list of dicts with 'doc' key containing full doc (must have 'text').
    Adds 'rerank_score' to each returned dict.
    """
    pairs = [(query, hit["doc"]["text"]) for hit in candidates]
    scores = reranker.predict(pairs)
    scored = [{**hit, "rerank_score": float(s)} for hit, s in zip(candidates, scores)]
    scored.sort(key=lambda h: h["rerank_score"], reverse=True)
    return scored[:top_k]

# Sanity check — score a single (query, doc) pair from the corpus
query = "what's error code AC-1042 about"
candidate_correct = {"id": "err_ac1042", "doc": next(d for d in corpus if d["id"] == "err_ac1042")}
candidate_wrong   = {"id": "err_ac4408", "doc": next(d for d in corpus if d["id"] == "err_ac4408")}

scored = rerank(query, [candidate_correct, candidate_wrong], top_k=2)
print(f"Query: {query!r}\n")
print(f"Reranker scores (higher = more relevant):")
for hit in scored:
    print(f"  {hit['id']:15s}  score = {hit['rerank_score']:+.3f}")
print()
print("The cross-encoder correctly ranks the ABOUT-AC-1042 doc over the")
print("MENTIONS-AC-1042 doc — the semantic distinction BM25 alone missed.")


#5 Full stack - hybrid + rerank on all 6 queries
# Rebuild BM25 + Qdrant setup (Day 1's variables aren't in scope in Day 2)

openai_client = OpenAI()
qdrant = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
COLLECTION = "wk09_day2_acme"

try: qdrant.delete_collection(COLLECTION)
except Exception: pass

qdrant.create_collection(collection_name=COLLECTION,
                        vectors_config=VectorParams(size=1536, distance=Distance.COSINE))

# Embed + upsert
texts = [d["text"] for d in corpus]
vectors = [item.embedding for item in openai_client.embeddings.create(
    model="text-embedding-3-small", input=texts).data]
qdrant.upsert(collection_name=COLLECTION, points=[
    PointStruct(id=i, vector=v, payload=d) for i, (d, v) in enumerate(zip(corpus, vectors))
])


# Build BM25
bm25 = build_bm25_index(corpus)
def dense_search(query: str, k: int = 10) -> list[dict]:
    q_vec = openai_client.embeddings.create(
        model="text-embedding-3-small", input=[query]).data[0].embedding
    hits = qdrant.query_points(collection_name=COLLECTION, query=q_vec, limit=k).points
    return [{"id": h.payload["id"], "title": h.payload["title"], "score": h.score,
             "doc": h.payload} for h in hits]

def hybrid_retrieve(query: str, k_per_retriever: int = 10, k_final: int = 10) -> list[dict]:
    bm25_hits  = bm25_search(bm25, corpus, query, k=k_per_retriever)
    dense_hits = dense_search(query, k=k_per_retriever)
    return rrf_fuse([bm25_hits, dense_hits], k=60, top_n=k_final)

print("✓ Full stack ready: hybrid_retrieve → rerank")
print(f"══ Full stack (hybrid → rerank) on all 6 queries (top-1) ══\n")
print(f"  {'Query':<50s}  {'Hybrid top-1':<18s}  {'Reranked top-1':<18s}  Verdict")
print(f"  {'-----':<50s}  {'------------':<18s}  {'--------------':<18s}  -------")

hybrid_correct   = 0
reranked_correct = 0

for tq in queries:
    hybrid_top10 = hybrid_retrieve(tq["q"], k_per_retriever=10, k_final=10)
    reranked_top3 = rerank(tq["q"], hybrid_top10, top_k=3)
    
    hybrid_top1 = hybrid_top10[0]["id"]
    rerank_top1 = reranked_top3[0]["id"]
    
    hybrid_ok   = hybrid_top1 == tq["expected_id"]
    rerank_ok   = rerank_top1 == tq["expected_id"]
    if hybrid_ok:   hybrid_correct += 1
    if rerank_ok:   reranked_correct += 1
    
    change = ""
    if not hybrid_ok and rerank_ok:  change = "  ← rerank FIXED"
    elif hybrid_ok and not rerank_ok: change = "  ← rerank BROKE (rare)"
    
    verdict = f"{'✓' if rerank_ok else '✗'} (expected {tq['expected_id']})"
    print(f"  {tq['q'][:50]:<50s}  {hybrid_top1:<18s}  {rerank_top1:<18s}  {verdict}{change}")

print()
print(f"══ Progression across all 3 stages ══")
print(f"  Hybrid retrieval (top-1):    {hybrid_correct}/{len(queries)}")
print(f"  + Cross-encoder rerank:      {reranked_correct}/{len(queries)}")

# Time the full pipeline over 3 runs to get a stable-ish read
test_query = "how do I export a dashboard to PDF"
N_RUNS = 3

stage_times = {"embed_query": [], "bm25": [], "dense": [], "rrf": [], "rerank": []}

for _ in range(N_RUNS):
    # Stage: embed the query
    t0 = time.time()
    q_vec = openai_client.embeddings.create(
        model="text-embedding-3-small", input=[test_query]).data[0].embedding
    stage_times["embed_query"].append(time.time() - t0)
    
    # Stage: BM25 search
    t0 = time.time()
    bm25_hits = bm25_search(bm25, corpus, test_query, k=10)
    stage_times["bm25"].append(time.time() - t0)
    
    # Stage: Qdrant dense search
    t0 = time.time()
    dense_hits = qdrant.query_points(collection_name=COLLECTION, query=q_vec, limit=10).points
    dense_hits = [{"id": h.payload["id"], "score": h.score, "doc": h.payload} for h in dense_hits]
    stage_times["dense"].append(time.time() - t0)
    
    # Stage: RRF fusion
    t0 = time.time()
    fused = rrf_fuse([bm25_hits, dense_hits], k=60, top_n=10)
    stage_times["rrf"].append(time.time() - t0)
    
    # Stage: cross-encoder rerank
    t0 = time.time()
    _ = rerank(test_query, fused, top_k=3)
    stage_times["rerank"].append(time.time() - t0)

print(f"══ Latency per stage (median of {N_RUNS} runs) ══\n")
print(f"  {'Stage':<15s}  {'Time (ms)':>10s}  Notes")
print(f"  {'-----':<15s}  {'---------':>10s}  -----")

def median(lst): return sorted(lst)[len(lst)//2]

notes = {
    "embed_query": "1 API call to OpenAI (network-bound)",
    "bm25":        "in-memory over 15 docs — trivial",
    "dense":       "1 API call to Qdrant (network-bound)",
    "rrf":         "pure Python arithmetic — free",
    "rerank":      "10 forward passes through CrossEncoder — CPU-bound",
}

total = 0
for stage, times in stage_times.items():
    ms = median(times) * 1000
    total += ms
    print(f"  {stage:<15s}  {ms:>8.1f}    {notes[stage]}")

print(f"  {'-'*15}  {'-'*9}")
print(f"  {'TOTAL':<15s}  {total:>8.1f} ms")

print()
print("Key observation: reranker is usually the slowest stage. Its cost scales")
print("LINEARLY with k_candidates. Doubling from 10 to 20 candidates ~doubles")
print("rerank time. That's why we pick k_candidates=10 (not 100) as default.")

#7 HYDE Query Rewriting #1: HyDE (Hypotheticl Document Embeddings)

def hyde_retrieve(query: str, k: int = 3) -> tuple[str, list[dict]]:
    """HyDE pattern: draft a fake answer, embed it, retrieve using it.
    Returns (hypothetical_answer, retrieved_docs)."""
    # 1. Ask the LLM for a hypothetical answer
    hyde_prompt = (
        f"Write a plausible one-paragraph answer to this question about a data "
        f"analytics platform. Don't hedge; write as if you know. If the question "
        f"asks about a specific product, feature, or error code, invent plausible "
        f"details.\n\nQuestion: {query}\n\nHypothetical answer:"
    )
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[{"role": "user", "content": hyde_prompt}],
    )
    hypothetical = resp.choices[0].message.content
    
    # 2. Embed the hypothetical and retrieve using it
    hyde_vec = openai_client.embeddings.create(
        model="text-embedding-3-small", input=[hypothetical]).data[0].embedding
    
    hits = qdrant.query_points(collection_name=COLLECTION, query=hyde_vec, limit=k).points
    docs = [{"id": h.payload["id"], "title": h.payload["title"], "score": h.score} for h in hits]
    return hypothetical, docs

# Demo on a query where dense alone struggled
test_q = "how do I get notified when something breaks"
print(f"Query: {test_q!r}\n")

# Baseline: plain dense retrieval
baseline = dense_search(test_q, k=3)
print(f"Baseline dense (top-3): {[h['id'] for h in baseline]}\n")

# HyDE
hypothetical, hyde_hits = hyde_retrieve(test_q, k=3)
print(f"LLM-drafted hypothetical answer:\n  {hypothetical}\n")
print(f"HyDE-retrieved (top-3): {[h['id'] for h in hyde_hits]}")

#8 Query Rewriting : Multi Query
def multi_query_retrieve(query: str, n_rewrites: int = 3, k: int = 5) -> tuple[list[str], list[dict]]:
    """Multi-query pattern: LLM rewrites → retrieve for each → RRF-fuse the ranked lists."""
    # 1. Get N rewrites from the LLM
    rewrite_prompt = (
        f"Rewrite this question in {n_rewrites} different ways that would return the same "
        f"answer. Vary terminology, sentence structure, and formality. Return each "
        f"rewrite on its own line — nothing else, no numbering.\n\nQuestion: {query}"
    )
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[{"role": "user", "content": rewrite_prompt}],
    )
    rewrites = [line.strip() for line in resp.choices[0].message.content.split("\n") if line.strip()]
    rewrites = rewrites[:n_rewrites]  # trim in case LLM over-produced
    
    # 2. Retrieve for each rewrite (also include the original query)
    all_queries = [query] + rewrites
    ranked_lists = [dense_search(q, k=k) for q in all_queries]
    
    # 3. RRF-fuse the ranked lists
    fused = rrf_fuse(ranked_lists, k=60, top_n=3)
    return rewrites, fused

# Demo on a query that could be phrased many ways
test_q = "my query is running slowly"
print(f"Query: {test_q!r}\n")

baseline = dense_search(test_q, k=3)
print(f"Baseline dense (top-3): {[h['id'] for h in baseline]}\n")

rewrites, mq_hits = multi_query_retrieve(test_q, n_rewrites=3, k=5)
print(f"LLM rewrites:")
for i, r in enumerate(rewrites, 1):
    print(f"  {i}. {r}")
print(f"\nMulti-query fused (top-3): {[h['id'] for h in mq_hits]}")

#9,10 Llama index 
llamaindex_sketch = '''
# ── LlamaIndex equivalent of our hybrid+rerank pipeline (~20 lines) ──
# NOT RUN — sketch only. Would require: pip install llama-index

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore

# 1. Set up dense retriever (Qdrant)
vector_store = QdrantVectorStore(client=qdrant_client, collection_name="capstone_chunks_v2")
index = VectorStoreIndex.from_vector_store(vector_store)
dense_retriever = index.as_retriever(similarity_top_k=10)

# 2. Set up BM25 retriever
bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)

# 3. Fuse them via LlamaIndex\'s built-in RRF
hybrid_retriever = QueryFusionRetriever(
    [dense_retriever, bm25_retriever],
    similarity_top_k=10,
    mode="reciprocal_rerank",  # RRF
)

# 4. Cross-encoder reranker as a postprocessor
reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=3)

# 5. Use it
candidates = hybrid_retriever.retrieve("how do I export a dashboard")
final = reranker.postprocess_nodes(candidates, query_str="how do I export a dashboard")
for node in final:
    print(node.text[:100], node.score)
'''

print(llamaindex_sketch)
print("\nWhat it does: exactly what we built in Day 1 + Day 2 Cells 1-5.")
print("What it hides: RRF math, tokenization, cross-encoder wrapping.")
print("What you gain: composition. Swap dense_retriever, bm25_retriever, or reranker")
print("               freely without touching the fusion or postprocessing.")
print("What you lose: visibility into the wire format between stages.")


#10 Clean up 
try:
    qdrant.delete_collection(COLLECTION)
    print(f"✓ Cleaned up demo collection: {COLLECTION}")
except Exception as e:
    print(f"Skip cleanup: {e}")
    