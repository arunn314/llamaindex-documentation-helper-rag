from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

PDF_PATH = "./sorcerers_stone.pdf"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZES = [2048, 512, 256, 128]
RERANK_TOP_N = 3

# Configure global settings
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

# Load document
documents = SimpleDirectoryReader(input_files=[PDF_PATH]).load_data()

# Build hierarchical nodes across all chunk sizes
node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=CHUNK_SIZES)
nodes = node_parser.get_nodes_from_documents(documents)

# Only leaf nodes go into the vector index; all nodes go into the docstore
leaf_nodes = get_leaf_nodes(nodes)

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

storage_context = StorageContext.from_defaults(docstore=docstore)

# Build index on leaf nodes only
index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)

# AutoMergingRetriever merges leaf nodes into parent when enough siblings are retrieved
base_retriever = index.as_retriever(similarity_top_k=12)
retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)

# Postprocessor: rerank merged results
postprocessors = [
    SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2",
        top_n=RERANK_TOP_N,
    ),
]

query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    node_postprocessors=postprocessors,
)


def query(question: str) -> str:
    response = query_engine.query(question)
    return str(response)


if __name__ == "__main__":
    questions = [
        "Who is Harry Potter?",
        "What is the Sorcerer's Stone?",
        "Who is Voldemort?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {query(q)}")
