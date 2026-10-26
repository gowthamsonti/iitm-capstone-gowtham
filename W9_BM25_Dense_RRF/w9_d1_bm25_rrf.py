import os
import sys
import json
import time
import re
from rank_bm25 import BM25Okapi
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# Load key-value pairs from the .env file into os.environ
load_dotenv(override=True)

assert os.environ.get("OPENAI_API_KEY") 
assert os.environ.get("QDRANT_URL")
assert os.environ.get("QDRANT_API_KEY")

# Add cwd to path so we can import the helper module
sys.path.insert(0, str(Path.cwd()))

# Load the corpus (15 docs) and 6 test queries with expected-winner annotations
CORPUS_PATH  = Path("sample_docs/acme_docs.jsonl")
QUERIES_PATH = Path("sample_docs/acme_test_queries.jsonl")
assert CORPUS_PATH.exists(), f"Missing {CORPUS_PATH}. Run: python demos/generate_acme_corpus.py"
assert QUERIES_PATH.exists(), f"Missing {QUERIES_PATH}. Run: python demos/generate_acme_corpus.py"

corpus  = [json.loads(line) for line in CORPUS_PATH.read_text().splitlines() if line]
queries = [json.loads(line) for line in QUERIES_PATH.read_text().splitlines() if line]

print(f"Loaded {len(corpus)} docs from {CORPUS_PATH.name}:")
from collections import Counter
for cat, n in Counter(d['category'] for d in corpus).items():
    print(f"  {cat:10s}  {n} docs")
print()
print(f"Loaded {len(queries)} test queries:")
for q in queries:
    print(f"  ({q['expected_winner']:5s})  {q['q']}")


openai_client = OpenAI()
qdrant = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])

COLLECTION = "wk09_day1_acme"

# Recreate — makes this cell re-runnable
try:
    qdrant.delete_collection(COLLECTION)
except Exception:
    pass

qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Embed all 15 docs in one API call
print("Embedding 15 Acme docs...")
texts = [d["text"] for d in corpus]
resp = openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
vectors = [item.embedding for item in resp.data]

# Upsert
points = [
    PointStruct(id=idx, vector=vec, payload=doc)
    for idx, (doc, vec) in enumerate(zip(corpus, vectors))
]

qdrant.upsert(collection_name=COLLECTION, points=points)
print(f"Upserted {len(points)} vectors into {COLLECTION!r}.")

def dense_search(query: str, k: int = 3) -> list[dict]:
    """Query Qdrant for top-K by cosine similarity."""
    q_vec = openai_client.embeddings.create(
        model="text-embedding-3-small", input=[query]
    ).data[0].embedding
    hits = qdrant.query_points(collection_name=COLLECTION, query=q_vec, limit=k).points
    return [
        {"id": h.payload["id"], "title": h.payload["title"], "score": h.score, "doc": h.payload}
        for h in hits
    ]

print(f"══ Dense retrieval on all 6 queries (top-1) ══\n")
print(f"  {'Expected':10s}  {'Query':<50s}  {'Retrieved':<20s}  Verdict")
print(f"  {'--------':10s}  {'-----':<50s}  {'---------':<20s}  -------")
dense_correct = 0

for tq in queries:
    hits = dense_search(tq["q"], k=1)
    got = hits[0]["id"]
    correct = got == tq["expected_id"]
    if correct: dense_correct += 1
    verdict = "✓" if correct else "✗ (expected " + tq["expected_id"] + ")"
    print(f"  {tq['expected_winner']:10s}  {tq['q'][:50]:<50s}  {got:<20s}  {verdict}")

print(f"\nDense-only: {dense_correct}/{len(queries)} correct at top-1")


#4 BM25 Fundementals

def simple_tokenize(text: str) -> list[str]:
    """Lowercase + word-and-alphanumeric split, KEEPING hyphens/slashes inside tokens.
    
    Critical: this pattern preserves 'ac-1042' and 'v2/dashboards' as single
    tokens rather than splitting them. That's what makes BM25 catch exact IDs.
    """
    return re.findall(r'[a-z0-9][a-z0-9\-/_]*', text.lower())

# Tokenization sanity check — critical to inspect before trusting BM25
print("══ Tokenization checks ══")
for text in ["Error AC-1042 indicates a timeout", "POST /v2/dashboards endpoint",
             "the enterprise plan cost"]:
    tokens = simple_tokenize(text)
    print(f"  {text!r:40s} → {tokens}")

# Fit BM25
print(f"\n══ Building BM25 index ══")
tokenized_corpus = [simple_tokenize(d["text"] + " " + d["title"]) for d in corpus]
bm25 = BM25Okapi(tokenized_corpus)
print(f"Indexed {len(corpus)} docs. Ready to query.")

def bm25_search(query: str, k: int = 3) -> list[dict]:
    """Query BM25 index; return top-K."""
    tokens = simple_tokenize(query)
    scores = bm25.get_scores(tokens)
    ranked = sorted(zip(scores, corpus), key=lambda pair: pair[0], reverse=True)
    return [
        {"id": doc["id"], "title": doc["title"], "score": float(score), "doc": doc}
        for score, doc in ranked[:k]
    ]

print(f"══ BM25 retrieval on all 6 queries (top-1) ══\n")
print(f"  {'Expected':10s}  {'Query':<50s}  {'Retrieved':<20s}  Verdict")
print(f"  {'--------':10s}  {'-----':<50s}  {'---------':<20s}  -------")

bm25_correct = 0
for tq in queries:
    hits = bm25_search(tq["q"], k=1)
    got = hits[0]["id"]
    correct = got == tq["expected_id"]
    if correct: bm25_correct += 1
    verdict = "✓" if correct else "✗ (expected " + tq["expected_id"] + ")"
    print(f"  {tq['expected_winner']:10s}  {tq['q'][:50]:<50s}  {got:<20s}  {verdict}")

print(f"\nBM25-only: {bm25_correct}/{len(queries)} correct at top-1")

#6 Neither dominates: Side by side comparism
print(f"══ Dense vs BM25 side-by-side (top-1 each) ══\n")
print(f"  {'Query':<50s}  {'DENSE top-1':<20s}  {'BM25 top-1':<20s}  {'Correct':<10s}")
print(f"  {'-----':<50s}  {'-----------':<20s}  {'----------':<20s}  {'-------':<10s}")

dense_wins, bm25_wins, both_win, both_fail = 0, 0, 0, 0

for tq in queries:
    dense_top = dense_search(tq["q"], k=1)[0]
    bm25_top  = bm25_search(tq["q"], k=1)[0]
    
    dense_ok = dense_top["id"] == tq["expected_id"]
    bm25_ok  = bm25_top["id"]  == tq["expected_id"]
    
    if dense_ok and bm25_ok:     tag = "both ✓";     both_win += 1
    elif dense_ok:               tag = "dense only";  dense_wins += 1
    elif bm25_ok:                tag = "bm25 only";   bm25_wins += 1
    else:                        tag = "both ✗";     both_fail += 1
    
    d_mark = "✓" if dense_ok else " "
    b_mark = "✓" if bm25_ok else " "
    print(f"  {tq['q'][:50]:<50s}  {d_mark} {dense_top['id']:<18s}  {b_mark} {bm25_top['id']:<18s}  {tag}")

print()
print(f"Both right:         {both_win}/{len(queries)}")
print(f"Only dense right:   {dense_wins}/{len(queries)}")
print(f"Only BM25 right:    {bm25_wins}/{len(queries)}")
print(f"Both wrong:         {both_fail}/{len(queries)}")


#7 RRF: How to contribute 2 rank lists
def rrf_fuse(ranked_lists: list[list[dict]], k: int = 60, top_n: int = 5) -> list[dict]:
    """Fuse multiple ranked lists via Reciprocal Rank Fusion.
    
    Each entry in ranked_lists is a list of dicts with an 'id' key.
    Returns the top-N fused results (with 'rrf_score' added).
    """
    scores = {}
    docs   = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            doc_id = hit["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in docs:
                docs[doc_id] = hit
    
    fused = sorted(scores.items(), key=lambda p: p[1], reverse=True)[:top_n]
    return [{**docs[doc_id], "rrf_score": score} for doc_id, score in fused]

# Toy example to show RRF math
print("══ RRF sanity check ══")
list_bm25  = [{"id": "doc_A"}, {"id": "doc_B"}, {"id": "doc_C"}]
list_dense = [{"id": "doc_C"}, {"id": "doc_A"}, {"id": "doc_D"}]

print(f"  BM25 ranks:  {[h['id'] for h in list_bm25]}")
print(f"  Dense ranks: {[h['id'] for h in list_dense]}")

fused = rrf_fuse([list_bm25, list_dense], k=60, top_n=5)
print(f"\n  RRF fused (k=60):")
for h in fused:
    print(f"    {h['id']}  rrf_score = {h['rrf_score']:.5f}")

print(f"\n  Interpretation:")
print(f"  - doc_A ranks 1st and 2nd → score = 1/61 + 1/62 = {1/61 + 1/62:.5f}")
print(f"  - doc_C ranks 3rd and 1st → score = 1/63 + 1/61 = {1/63 + 1/61:.5f}")
print(f"  - doc_B appears in one list only → score = 1/62 = {1/62:.5f}")
print(f"  - doc_D appears in one list only → score = 1/63 = {1/63:.5f}")

def hybrid_retrieve(query: str, k_per_retriever: int = 10, k_final: int = 3) -> list[dict]:
    """BM25 + Dense + RRF combined retrieval.
    
    k_per_retriever: how many to take from each retriever before fusion (10 is standard)
    k_final:         how many to return after RRF fusion
    """
    bm25_hits  = bm25_search(query, k=k_per_retriever)
    dense_hits = dense_search(query, k=k_per_retriever)
    return rrf_fuse([bm25_hits, dense_hits], k=60, top_n=k_final)

print(f"══ Hybrid retrieval on all 6 queries (top-1) ══\n")
print(f"  {'Query':<50s}  {'Hybrid top-1':<20s}  {'Verdict'}")
print(f"  {'-----':<50s}  {'------------':<20s}  {'-------'}")

hybrid_correct = 0
for tq in queries:
    hits = hybrid_retrieve(tq["q"], k_per_retriever=10, k_final=1)
    got = hits[0]["id"]
    correct = got == tq["expected_id"]
    if correct: hybrid_correct += 1
    verdict = "✓" if correct else "✗ (expected " + tq["expected_id"] + ")"
    print(f"  {tq['q'][:50]:<50s}  {got:<20s}  {verdict}")

print()
print(f"══ Summary across the 3 retrievers ══")
print(f"  Dense only:  {dense_correct}/{len(queries)}")
print(f"  BM25 only:   {bm25_correct}/{len(queries)}")
print(f"  Hybrid RRF:  {hybrid_correct}/{len(queries)}")
