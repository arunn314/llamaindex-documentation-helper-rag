from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

PDF_PATH = "./sorcerers_stone.pdf"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
WINDOW_SIZE = 3
RERANK_TOP_N = 3

# Configure global settings
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

# Load document
documents = SimpleDirectoryReader(input_files=[PDF_PATH]).load_data()

# Build sentence window nodes
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=WINDOW_SIZE,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# Build index
index = VectorStoreIndex.from_documents(
    documents,
    transformations=[node_parser],
)

# Postprocessors: replace sentence with window, then rerank
postprocessors = [
    MetadataReplacementPostProcessor(target_metadata_key="window"),
    SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2",
        top_n=RERANK_TOP_N,
    ),
]

# Query engine
query_engine = index.as_query_engine(
    similarity_top_k=6,
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
