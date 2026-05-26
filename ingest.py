import time
import chromadb

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "./llamaindex-docs"
CHROMA_DIR = "./chroma_db"
CHROMA_COLLECTION = "llamaindex_docs"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
NUM_DOCS = 1200
TEST_QUERY = "How do I create a VectorStoreIndex in LlamaIndex?"

# ── Chunk quality analysis ────────────────────────────────────────────────────
def analyze_chunk_quality(nodes):
    buckets = {"small (0-250)": 0, "medium (251-500)": 0, "large (501-750)": 0, "xlarge (751+)": 0}

    lengths = [len(node.get_content()) for node in nodes]
    for length in lengths:
        if length <= 250:
            buckets["small (0-250)"] += 1
        elif length <= 500:
            buckets["medium (251-500)"] += 1
        elif length <= 750:
            buckets["large (501-750)"] += 1
        else:
            buckets["xlarge (751+)"] += 1

    total = len(lengths)
    avg = sum(lengths) / total if total else 0

    print("\nChunk Quality Analysis")
    print("=" * 40)
    print(f"  Total chunks : {total}")
    print(f"  Avg length   : {avg:.0f} chars")
    print(f"  Min / Max    : {min(lengths)} / {max(lengths)} chars")
    print()
    bar_width = 20
    for label, count in buckets.items():
        pct = count / total * 100 if total else 0
        bar = "█" * int(pct / 100 * bar_width)
        print(f"  {label:<20} {bar:<{bar_width}} {count:>4} chunks ({pct:5.1f}%)")
    print("=" * 40)


# ── Models ────────────────────────────────────────────────────────────────────
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
llm = Ollama(model="llama3.2", request_timeout=120.0)

# ── Load documents ────────────────────────────────────────────────────────────
print(f"Loading {NUM_DOCS} documents from '{DATA_DIR}' ...")
documents = SimpleDirectoryReader(DATA_DIR, num_files_limit=NUM_DOCS).load_data()
print(f"  Loaded {len(documents)} documents.")

# ── ChromaDB vector store ─────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
if CHROMA_COLLECTION in [c.name for c in chroma_client.list_collections()]:
    chroma_client.delete_collection(CHROMA_COLLECTION)
    print(f"  Deleted existing collection '{CHROMA_COLLECTION}'.")
chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# ── Ingestion pipeline ────────────────────────────────────────────────────────
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
        embed_model,
    ],
)

print("Running ingestion pipeline ...")
start = time.time()
nodes = pipeline.run(documents=documents, show_progress=True)
elapsed = time.time() - start
print(f"  Processed {len(nodes)} nodes in {elapsed:.2f}s.")

print("Adding nodes to ChromaDB in batches ...")
max_batch = chroma_client.get_max_batch_size()
for i in range(0, len(nodes), max_batch):
    vector_store.add(nodes[i : i + max_batch])
print(f"  Ingested {len(nodes)} nodes.")

analyze_chunk_quality(nodes)

# ── Build index from the populated vector store ───────────────────────────────
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
)

# ── Test query ────────────────────────────────────────────────────────────────
query_engine = index.as_query_engine(llm=llm)
print(f"\nTest query: '{TEST_QUERY}'")
print("-" * 60)
response = query_engine.query(TEST_QUERY)
print(response)
