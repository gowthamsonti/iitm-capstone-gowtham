import os
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, PointStruct
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType

# Load key-value pairs from the .env file into os.environ
load_dotenv(override=True)

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY before running this notebook"
assert os.environ.get("QDRANT_URL"),     "Set QDRANT_URL — get free-tier at cloud.qdrant.io"
assert os.environ.get("QDRANT_API_KEY"), "Set QDRANT_API_KEY — from your Qdrant Cloud cluster"

from wk07_pipeline import (
    CORPUS_ANIMALS,
    TEST_QUESTIONS,
    embed_batch,
    embed_one,
    cosine, dot, l2,
    get_qdrant_client,
    EMBED_SMALL,
) 

openai_client = OpenAI()

print(f"Loaded {len(CORPUS_ANIMALS)} animals from Day 1's helper.")
print(f"Loaded {len(TEST_QUESTIONS)} test questions.")
print("Metrics available: cosine, dot, l2")
print("Pipeline functions: embed_batch, embed_one, get_qdrant_client")

#2 Reconnect to qdrant and embed to corpus

qdrant = get_qdrant_client()
# Sanity: what collections already exist?
existing = [c.name for c in qdrant.get_collections().collections]
print(f"Existing collections: {existing}\n")

# Embed all 10 animals fresh (Day 1's variables aren't in scope here)
print("Embedding animals with text-embedding-3-small...")
texts = [d["text"] for d in CORPUS_ANIMALS]
vectors = embed_batch(texts, model=EMBED_SMALL)

# Attach to records
for doc, vec in zip(CORPUS_ANIMALS, vectors):
    doc["vector"] = vec

print(f"Embedded {len(vectors)} animals ({len(vectors[0])} dims each).")

#3 Problem indices solve:

def linear_scan(query_vec, corpus, k=3):
    """Naive linear scan — compute similarity vs every doc, sort, take top-K."""
    scored = [(cosine(query_vec, doc["vector"]), doc) for doc in corpus]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:k]


query = "Which animals live in the ocean?"
q_vec = embed_one(query)

# Time the linear scan on our 10-doc corpus
t0 = time.time()
hits = linear_scan(q_vec, CORPUS_ANIMALS, k=3)
dt = time.time() - t0


print(f"Q: {query!r}")
print(f"Linear scan over {len(CORPUS_ANIMALS)} docs took {dt*1000:.2f} ms\n")
for score, doc in hits:
    print(f"  {score:.3f}  {doc['id']:8s}  ({doc['category']})")


# Now simulate a bigger corpus to see how linear scan scales.
# We fake it by copying our 10 vectors many times.

print(f"  {'Corpus size':>12s}  {'Linear scan time':>18s}  {'ops (approx)':>15s}")
print(f"  {'-----------':>12s}  {'----------------':>18s}  {'-----------':>15s}")

for size in [10, 100, 1_000, 10_000]:
    # Build a fake corpus by tiling our 10 vectors
    fake_corpus = [{"vector": CORPUS_ANIMALS[i % 10]["vector"],
                    "id": f"fake_{i}"}
                   for i in range(size)]
    
    t0 = time.time()
    scored = [(cosine(q_vec, d["vector"]), d) for d in fake_corpus]
    scored.sort(key=lambda x: x[0], reverse=True)
    _ = scored[:3]
    dt = time.time() - t0
    
    ops = size * 1536
    print(f"  {size:>12,d}  {dt*1000:>15.1f} ms  {ops:>15,d}")


print()
print("At 10K docs, linear scan is still workable (~10s).")
print("At 100K, it's painful. At 10M — completely impractical.")
print("\nSolution: pre-build an *index* that skips most of the work at query time.")


#4,5 Indices trade off
#HNSW - Graph Based Qdrant default, IVF - Cluster Based.- Fast build good for large corpus


def recreate_collection(name, size=1536, distance=Distance.COSINE, hnsw=None):
    """Delete-then-create so cells are re-runnable."""
    try:
        qdrant.delete_collection(name)
    except Exception:
        pass
    qdrant.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=size, distance=distance),
        hnsw_config=hnsw,
    )


def upsert_animals(collection_name):
    """Push the 10 animals into a collection."""
    points = [
        PointStruct(
            id=idx,
            vector=doc["vector"],
            payload={"animal_id": doc["id"], "category": doc["category"], "text": doc["text"]},
        )
        for idx, doc in enumerate(CORPUS_ANIMALS)
    ]
    qdrant.upsert(collection_name=collection_name, points=points)


COLL_HNSW_DEFAULT = "wk07_day2_hnsw_default"
recreate_collection(COLL_HNSW_DEFAULT)  # HNSW is Qdrant's default; no config = defaults
upsert_animals(COLL_HNSW_DEFAULT)
# Show the default HNSW parameters Qdrant used
info = qdrant.get_collection(COLL_HNSW_DEFAULT)
hnsw = info.config.hnsw_config

print(f"Collection {COLL_HNSW_DEFAULT!r} — HNSW default config:")
print(f"  m:               {hnsw.m}          (graph connectivity — higher = more edges per node)")
print(f"  ef_construct:    {hnsw.ef_construct}         (build-time search depth — higher = better recall, slower build)")
print(f"  full_scan_threshold: {hnsw.full_scan_threshold}   (below this many points, use linear scan)")

#6 HNSW Tuned for Higher Recall:

COLL_HNSW_TUNED = "wk07_day2_hnsw_tuned"
recreate_collection(
    COLL_HNSW_TUNED,
    hnsw=HnswConfigDiff(m=32, ef_construct=200),  # 2× default on both
)
upsert_animals(COLL_HNSW_TUNED)

info = qdrant.get_collection(COLL_HNSW_TUNED)
hnsw = info.config.hnsw_config
print(f"Collection {COLL_HNSW_TUNED!r} — HNSW tuned config:")
print(f"  m:            {hnsw.m}  (was 16, now 32)")
print(f"  ef_construct: {hnsw.ef_construct}  (was 100, now 200)")
print()
print("Effect at production scale:")
print("  - Better recall (misses fewer true top-K on approximate queries)")
print("  - More memory (each node has 2× the edges to store)")
print("  - Slower build time (index construction does 2× more work)")
print("  - Query speed roughly the same (HNSW query cost is O(log n × ef))")

#7 Query both collections results should match

query = "Which animals live in the ocean?"
q_vec = embed_one(query)

for coll in [COLL_HNSW_DEFAULT, COLL_HNSW_TUNED]:
    hits = qdrant.query_points(collection_name=coll, query=q_vec, limit=3).points
    print(f"  {coll}:")
    for h in hits:
        p = h.payload
        print(f"    {h.score:.3f}  {p['animal_id']:8s} ({p['category']})")
    print()

#8 Qdrant quantization(IVF's Philosophical Cousin)

# IVF Inverted File Index 

COLL_QUANT = "wk07_day2_quantized"
try:
    qdrant.delete_collection(COLL_QUANT)
except Exception:
    pass

qdrant.create_collection(
    collection_name=COLL_QUANT,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        )
    ),
)
upsert_animals(COLL_QUANT)

info = qdrant.get_collection(COLL_QUANT)
print(f"Collection {COLL_QUANT!r} created with scalar quantization.")
print(f"  Storage per vector before:  1536 × 4 bytes = 6144 bytes")
print(f"  Storage per vector after:   1536 × 1 byte  = 1536 bytes  (4× smaller)")
print()
print("On our 10-animal corpus this saves ~45 KB — trivial.")
print("On 10M vectors, this saves 45 GB. That's when quantization pays off.")


#9 Query the quantized collection:

query = "Which animals live in the ocean?"
q_vec = embed_one(query)

print(f"Q: {query!r}\n")
for coll in [COLL_HNSW_DEFAULT, COLL_QUANT]:
    hits = qdrant.query_points(collection_name=coll, query=q_vec, limit=3).points
    label = "FULL PRECISION" if coll == COLL_HNSW_DEFAULT else "QUANTIZED (int8)"
    print(f"  {label}:")
    for h in hits:
        p = h.payload
        print(f"    {h.score:.3f}  {p['animal_id']:8s} ({p['category']})")
    print()

#11 Metrics Range Compared

vec_cat   = next(d['vector'] for d in CORPUS_ANIMALS if d['id'] == 'cat')
vec_dog   = next(d['vector'] for d in CORPUS_ANIMALS if d['id'] == 'dog')
vec_bee   = next(d['vector'] for d in CORPUS_ANIMALS if d['id'] == 'bee')

pairs = [("cat", "dog", vec_cat, vec_dog, "similar (both mammals)"),
         ("cat", "bee", vec_cat, vec_bee, "different (mammal vs insect)")]


print(f"  {'Pair':15s}  {'cosine':>8s}  {'dot':>8s}  {'l2':>8s}   Interpretation")
print(f"  {'----':15s}  {'------':>8s}  {'---':>8s}  {'--':>8s}   {'-'*30}")
for a, b, va, vb, note in pairs:
    c = cosine(va, vb)
    d = dot(va, vb)
    l = l2(va, vb)
    print(f"  {a:<5s} vs {b:6s}  {c:>8.3f}  {d:>8.3f}  {l:>8.3f}   {note}")

print()
print("Note: for OpenAI embeddings (unit-length), cosine ≈ dot product,")
print("and L2 = sqrt(2 - 2*cosine).")
print("Cosine higher = closer.  Dot higher = closer.  L2 lower = closer.")

#12, 13 

# Simulate non-normalised custom embeddings: scale some vectors up 3x for no reason
scaled_docs = []
for d in CORPUS_ANIMALS:
    scale = 3.0 if d['id'] in ["salmon", "ant"] else 1.0   # arbitrarily scaled
    vec = [v * scale for v in d['vector']]
    scaled_docs.append({**d, "vector": vec, "scale": scale})

# Query 'ocean animals' — the honest cosine ranking should put salmon and shark on top.
query = "Which animals live in the ocean?"
q_vec = embed_one(query)

# Rank by cosine (correct — scale-invariant)
cos_scored = [(cosine(q_vec, d['vector']), d) for d in scaled_docs]
cos_scored.sort(key=lambda p: p[0], reverse=True)

# Rank by dot product (WRONG when vectors aren't normalised — magnitude wins)
dot_scored = [(dot(q_vec, d['vector']), d) for d in scaled_docs]
dot_scored.sort(key=lambda p: p[0], reverse=True)

print(f"Q: {query!r}\n")
print("  COSINE ranking (correct):")
for s, d in cos_scored[:5]:
    marker = "  ← scaled 3x" if d['scale'] > 1 else ""
    print(f"    {s:>7.3f}  {d['id']:8s} ({d['category']:7s}){marker}")

print("\n  DOT PRODUCT ranking (buggy — magnitude wins over meaning):")
for s, d in dot_scored[:5]:
    marker = "  ← scaled 3x" if d['scale'] > 1 else ""
    print(f"    {s:>7.3f}  {d['id']:8s} ({d['category']:7s}){marker}")


#14 Mini rag on top of Qdrant:

SYSTEM = (
    "You are a helpful assistant. Answer using ONLY the provided context. "
    "If the context does not contain the answer, say so plainly. Cite the "
    "animal id in square brackets after any fact you use."
)


def ask_qdrant_rag(question: str, collection: str, k: int = 3) -> dict:
    """Full mini-RAG: embed → Qdrant retrieve → prompt → generate."""
    # 1. Embed the query
    q_vec = embed_one(question)
    # 2. Retrive Top -k from Qdrant
    hits = qdrant.query_points(collection_name=collection,
                               query=q_vec, limit = k).points

    # 3. Build the prompt with retrieved chunks

    context = "\n\n".join(
        f"[{h.payload['animal_id']}]\n{h.payload['text']}"
        for h in hits
    )
    print(f"context")

    # 4. Generate
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content":
                f"Context:\n{context}\n\n---\n\nQuestion: {question}"},
        ],
    )

    return {
        "question": question,
        "answer":   resp.choices[0].message.content,
        "sources":  [h.payload["animal_id"] for h in hits],
    }


# Test on one question
COLL_COSINE=COLL_HNSW_DEFAULT
result = ask_qdrant_rag("Which animals live in the ocean?", COLL_COSINE)
print(f"Q: {result['question']}\n")
print(f"A: {result['answer']}\n")
print(f"Sources: {result['sources']}")


for tq in TEST_QUESTIONS:
    result = ask_qdrant_rag(tq["q"], COLL_COSINE, k=3)
    print(f"── {tq['q']}")
    print(f"   expected categories: {tq['expected_categories']}")
    print(f"   sources retrieved:   {result['sources']}")
    print(f"   answer: {result['answer']}")
    print()
