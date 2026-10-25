import os
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

# Load key-value pairs from the .env file into os.environ
load_dotenv(override=True)

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY before running this notebook"
assert os.environ.get("QDRANT_URL"),     "Set QDRANT_URL — get free-tier at cloud.qdrant.io"
assert os.environ.get("QDRANT_API_KEY"), "Set QDRANT_API_KEY — from your Qdrant Cloud cluster"

client = OpenAI()

EMBED_SMALL = "text-embedding-3-small"    # 1536 dims, $0.02 per 1M tokens
EMBED_LARGE = "text-embedding-3-large"    # 3072 dims, $0.13 per 1M tokens

print("Setup ok.")
print(f"OpenAI key set.")
print(f"Qdrant URL: {os.environ['QDRANT_URL']}...")
print(f"Qdrant Key: {os.environ['QDRANT_API_KEY']}")
print(f"Openai Key: {os.environ['OPENAI_API_KEY']}")

#2 Corpus
documents = [
    # Mammals
    {"id": "cat",     "category": "mammal",  "text": (
        "The domestic cat is a small carnivorous mammal kept as a pet by humans "
        "for thousands of years. Cats have keen senses, retractable claws, and "
        "are known for their agility, night vision, and independent behaviour."
    )},
    {"id": "dog",     "category": "mammal",  "text": (
        "The dog is a domesticated carnivorous mammal descended from wolves. "
        "Dogs are highly social animals kept by humans for companionship, work, "
        "and protection. They are known for loyalty, trainability, and an acute "
        "sense of smell."
    )},
    # Birds
    {"id": "eagle",   "category": "bird",    "text": (
        "Eagles are large birds of prey with keen eyesight, powerful hooked beaks, "
        "and strong talons. They soar at high altitudes and hunt smaller animals "
        "including fish, rabbits, and reptiles."
    )},
    {"id": "sparrow", "category": "bird",    "text": (
        "Sparrows are small, plump songbirds found across most of the world. They "
        "typically have short beaks adapted for eating seeds and grain, though they "
        "also feed on insects."
    )},
    # Fish
    {"id": "salmon",  "category": "fish",    "text": (
        "Salmon are ray-finned fish found in the Northern Pacific and Atlantic oceans. "
        "They are anadromous — born in freshwater, migrating to the sea to mature, "
        "then returning upstream to spawn."
    )},
    {"id": "shark",   "category": "fish",    "text": (
        "Sharks are cartilaginous fish that have inhabited the oceans for over 400 "
        "million years. They have a keen sense of smell, multiple rows of "
        "regenerating teeth, and a hydrodynamic shape."
    )},
    # Reptiles
    {"id": "python",  "category": "reptile", "text": (
        "Pythons are non-venomous constrictor snakes native to Africa, Asia, and "
        "Australia. They kill prey by wrapping their muscular bodies around it and "
        "squeezing until circulation stops. Some species grow over six metres long."
    )},
    {"id": "gecko",   "category": "reptile", "text": (
        "Geckos are small lizards found in warm climates worldwide. They can climb "
        "smooth vertical surfaces using microscopic hairs on their toe pads. Many "
        "gecko species make chirping vocalisations."
    )},
    # Insects
    {"id": "bee",     "category": "insect",  "text": (
        "Honey bees are flying insects known for producing honey and beeswax. They "
        "live in colonies with a strict social structure. Bees pollinate a large "
        "fraction of the world's flowering plants."
    )},
    {"id": "ant",     "category": "insect",  "text": (
        "Ants are eusocial insects that live in large organised colonies. They "
        "communicate primarily through pheromones and are found on every continent "
        "except Antarctica."
    )},
]

print(f"Loaded {len(documents)} documents")

for d in documents:
    print(f" {d['id']:8s} ({d['category']:7s})  {len(d['text']) : 3d} chars")

# Embedding Recap
sample_sentence = "A domestic cat is a small carnivorous mammal."

def embed_one(text: str, model: str = EMBED_SMALL) -> list[float]:
    resp = client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding

vec = embed_one(sample_sentence)

print(f"Sentence: {sample_sentence!r}")
print(f"Model:    {EMBED_SMALL}")
print(f"Vector length (dim): {len(vec)}")
print(f"First 8 dims: {vec[:8]}")
print(f"Type of each element: {type(vec[0]).__name__}")
print("\nTakeaway: an embedding is just a list of floats. You cannot read it.")
print("But the model has arranged them so that similar meanings live nearby in this space.")


#4 Compare small vs large embedding model

# Embed the same sentence with both models
sentence = "Cats have keen night vision and retractable claws."

t0 = time.time()
vec_small = embed_one(sentence, model=EMBED_SMALL)
dt_small  = time.time() - t0

t0 = time.time()
vec_large = embed_one(sentence, model=EMBED_LARGE)
dt_large  = time.time() - t0

print(f"Sentence: {sentence!r}\n")
print(f"  {EMBED_SMALL}:")
print(f"    dim:     {len(vec_small)}")
print(f"    latency: {dt_small*1000:.0f} ms")
print(f"    first 5: {vec_small[:5]}\n")
print(f"  {EMBED_LARGE}:")
print(f"    dim:     {len(vec_large)}")
print(f"    latency: {dt_large*1000:.0f} ms")
print(f"    first 5: {vec_large[:5]}\n")
print("Same sentence, different vectors, different lengths. Both are 'right' —")
print("they're just different coordinate systems for the same meaning.")

#Does the large model discriminate better

# Experiment cosine between 3 pairs of animals
# Same category cat vs dog
# diff category cat vs eagle
# distant categories vat vs bee

def cosine(a,b):
    va, vb = np.array(a),np.array(b)
    return float(np.dot(va,vb)/np.linalg.norm(va)* np.linalg.norm(vb))

# Get the three animal texts
text_cat   = next(d['text'] for d in documents if d['id'] == 'cat')
text_dog   = next(d['text'] for d in documents if d['id'] == 'dog')
text_eagle = next(d['text'] for d in documents if d['id'] == 'eagle')
text_bee   = next(d['text'] for d in documents if d['id'] == 'bee')


# Embed all four with both models
e_small = {name: embed_one(txt, model=EMBED_SMALL)
           for name, txt in [("cat", text_cat), ("dog", text_dog),
                              ("eagle", text_eagle), ("bee", text_bee)]}
e_large = {name: embed_one(txt, model=EMBED_LARGE)
           for name, txt in [("cat", text_cat), ("dog", text_dog),
                              ("eagle", text_eagle), ("bee", text_bee)]}


pairs = [("cat", "dog",   "same category (both mammals)"),
         ("cat", "eagle", "related (both predators)"),
         ("cat", "bee",   "distant (mammal vs insect)")]

print(f"  {'Pair':15s}  {'3-small':>8s}  {'3-large':>8s}   Interpretation")
print(f"  {'----':15s}  {'-------':>8s}  {'-------':>8s}   {'-'*30}")

for a, b, note in pairs:
    s = cosine(e_small[a], e_small[b])
    l = cosine(e_large[a], e_large[b])
    print(f"  {a:<5s} vs {b:6s}  {s:>8.3f}  {l:>8.3f}   {note}")

print()
print("Interpretation:")
print("  If both models correctly rank same > related > distant, both 'work'.")
print("  If 3-large shows a WIDER spread between same and distant, it's more discriminative.")
print("  If the spreads are similar, 3-small is doing the job at 1/6.5 the cost.")



# Storage math for different corpus sizes
print(f"  {'Corpus':>15s}  {'3-small storage':>18s}  {'3-large storage':>18s}  Comment")
print(f"  {'------':>15s}  {'---------------':>18s}  {'---------------':>18s}  {'-'*30}")

for n_chunks in [10, 100, 1_000, 10_000, 100_000]:
    small_bytes = n_chunks * 1536 * 4   # 4 bytes per float32
    large_bytes = n_chunks * 3072 * 4
    small_mb    = small_bytes / (1024*1024)
    large_mb    = large_bytes / (1024*1024)
    if n_chunks == 10:
        comment = "our demo corpus"
    elif n_chunks == 100:
        comment = "typical W6 capstone"
    elif n_chunks == 10_000:
        comment = "medium enterprise"
    elif n_chunks == 100_000:
        comment = "large corpus — starts to matter"
    else:
        comment = ""
    print(f"  {n_chunks:>15,d}  {small_mb:>15.2f} MB  {large_mb:>15.2f} MB  {comment}")

print()
print("At our scale (< 1K chunks), storage difference is trivial. The cost you")
print("care about is the API call: 3-large is 6.5× 3-small per embedding.")
print("At 100K chunks, storage starts to matter — but you'd use a proper vector")
print("DB (like Qdrant, which we meet later today) that handles both scales.")


#7 Sementic Geomentry : Embed all 10 animals
def embed_batch(texts, model=EMBED_SMALL):
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]

# Embed all 10 animals in one API call
texts = [d["text"] for d in documents]
vectors = embed_batch(texts)

# Attach the vector to each animal record
for doc, vec in zip(documents, vectors):
    doc["vector"] = vec

print(f"Embedded {len(documents)} animals.\n")
for d in documents:
    print(f"  {d['id']:8s}  ({d['category']:7s})  dim={len(d['vector'])}")


#Pair wise cosine matrix

# Print the pairwise matrix as a formatted table
ids = [d["id"] for d in documents]
# Header row
print(f"  {'':8s}", end="")
for other_id in ids:
    print(f"{other_id[:6]:>7s}", end="")
print()

for i, d in enumerate(documents):
    print(f"  {d['id']:8s}", end="")
    for j, other in enumerate(documents):
        if i == j:
            print(f"{'--':>7s}", end="")  # self-similarity is 1.0, skip
        else:
            score = cosine(d["vector"], other["vector"])
            print(f"{score:>7.3f}", end="")
    print(f"  ({d['category']})")

print("\nRead across each row: which animals is this one most similar to?")
print("Look for block structure — do same-category pairs score higher than cross-category?")


#9 Nearest neiboours per animal
# for each animal print top 3 nearest animals

for d in documents:
    scored = [(cosine(d["vector"], other["vector"]), other)
              for other in documents if other["id"] != d["id"]]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    top3 = scored[:3]
    neighbours_str = ", ".join(f"{o['id']} ({o['category']}, {s:.3f})" for s, o in top3)
    print(f"  {d['id']:8s} ({d['category']:7s})  →  {neighbours_str}")
print()
print("How often is the top neighbour in the SAME category as the query?")
print("That's a rough measure of how 'semantically aligned' the model is with our labels.")


#Vector db landscape
# for shorter abouve is fine how about millions and billions of chunks

# A tiny lookup that prints the trade-offs for each option.
# Read them out loud; discuss which you'd pick for what.

VECTOR_DBS = {
    "FAISS": {
        "maker": "Meta AI Research",
        "type": "library (in-process)",
        "strengths": "fastest in-memory search; battle-tested at Meta scale",
        "weaknesses": "no persistence by default; no service layer; you manage everything",
        "best for": "batch pipelines, research, embedded use",
    },
    "Chroma": {
        "maker": "Chroma Inc.",
        "type": "embedded DB / service",
        "strengths": "easiest 5-minute setup; friendly Python-first API",
        "weaknesses": "younger project; less battle-tested at scale; smaller ecosystem",
        "best for": "prototypes, small apps, learning",
    },
    "Qdrant": {
        "maker": "Qdrant",
        "type": "service (Rust core, Python SDK)",
        "strengths": "production-grade; strong filtering; persistence built in; good free tier",
        "weaknesses": "needs a running service (Docker or Cloud); slight setup cost",
        "best for": "real applications; anything hitting production",
    },
    "pgvector": {
        "maker": "open-source (Andrew Kane)",
        "type": "PostgreSQL extension",
        "strengths": "reuses Postgres you may already have; JOIN with SQL tables",
        "weaknesses": "slower than dedicated vector DBs at scale; simpler indexing options",
        "best for": "teams already on Postgres; small-to-medium vector workloads",
    },
}

def describe_db(name):
    info = VECTOR_DBS[name]
    print(f"══ {name} ══")
    for key, val in info.items():
        print(f"  {key:11s}  {val}")
    print()

for name in VECTOR_DBS:
    describe_db(name)

#11 Connect to Qdrant
#signed up in cloud.qdrant.io
# setup QDRANT_URL and QDRANT_API_KEY in .env
# pip install qdrant-client

qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)


# Sanity check — list existing collections
existing = qdrant.get_collections()
print(f"Connected to Qdrant at {os.environ['QDRANT_URL'][:40]}...")
print(f"Existing collections: {[c.name for c in existing.collections]}")
print("\nIf you see [] (empty), that's fine — this is a fresh cluster.")

#12 Create a collection
COLLECTION_NAME = "wk07_day1_animals"

# Delete any prior version — makes this cell re-runnable
try:
    qdrant.delete_collection(COLLECTION_NAME)
    print(f"Deleted existing {COLLECTION_NAME!r} collection.")
except Exception:
    pass  # didn't exist yet

# Create fresh
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

info = qdrant.get_collection(COLLECTION_NAME)
print(f"Created collection {COLLECTION_NAME!r}")
print(f"  dim:      {info.config.params.vectors.size}")
print(f"  metric:   {info.config.params.vectors.distance}")
print(f"  points:   {info.points_count}")


#13 Upsert the animals

points = [
    PointStruct(
        id=idx,
        vector=doc["vector"],
        payload={
            "animal_id": doc["id"],
            "category":  doc["category"],
            "text":      doc["text"],
        },
    )
    for idx, doc in enumerate(documents)
]

qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
# Verify
info = qdrant.get_collection(COLLECTION_NAME)
print(f"Upserted {len(points)} points.")
print(f"Collection now has {info.points_count} points.")


#14 First Retrival

queries = [
    "Which animals live in the ocean?",
    "What insects live in social colonies?",
    "What animals hunt from the sky?",
    "Which animals are kept as pets?",
]

for query in queries:
    q_vec = embed_one(query)
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        limit=3,
    ).points

    print(f"Q: {query!r}")
    for i, hit in enumerate(results, 1):
        p = hit.payload
        print(f"  [{i}] score={hit.score:.3f}  {p['animal_id']:8s} ({p['category']})")
    print()