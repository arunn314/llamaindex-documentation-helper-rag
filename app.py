import streamlit as st
import chromadb

from typing import List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
CHROMA_COLLECTION = "llamaindex_docs"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOKEN_LIMIT = 3000
JACCARD_THRESHOLD = 0.85


# ── Jaccard deduplication post-processor ──────────────────────────────────────
class JaccardDeduplicator(BaseNodePostprocessor):
    """Remove near-duplicate nodes whose word-level Jaccard similarity
    exceeds the threshold against any already-kept node."""

    threshold: float = JACCARD_THRESHOLD

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        kept_word_sets: List[set] = []
        result: List[NodeWithScore] = []

        for node in nodes:
            words = set(node.get_content().lower().split())
            duplicate = False
            for kept in kept_word_sets:
                union = words | kept
                if union and len(words & kept) / len(union) >= self.threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept_word_sets.append(words)
                result.append(node)

        return result


# ── Cached resource: chat engine ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading index from ChromaDB ...")
def load_chat_engine():
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    llm = Ollama(model="llama3.2", request_timeout=120.0)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    chat_engine = index.as_chat_engine(
        llm=llm,
        memory=ChatMemoryBuffer.from_defaults(token_limit=TOKEN_LIMIT),
        node_postprocessors=[JaccardDeduplicator()],
        chat_mode="context",
        system_prompt=(
                "You are a helpful assistant that answers questions about LlamaIndex. "
                "Use the retrieved context to provide accurate, helpful answers. "
                "If you don't know the answer, say so."
            ),
        verbose=True,
    )
    return chat_engine


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="LlamaIndex Doc Helper", page_icon="🦙")
st.title("LlamaIndex Doc Helper with Chroma Vector Store")
st.caption("Ask questions about your documents stored in Chroma Vector Store")

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = load_chat_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []  # each entry: {role, content, sources?}

def render_sources(sources):
    if not sources:
        return
    with st.expander("Sources", expanded=False):
        for i, node in enumerate(sources[:2], 1):
            score = node.score if node.score is not None else 0.0
            doc_name = node.metadata.get("file_name", "unknown")
            preview = node.get_content()[:200].replace("\n", " ")
            st.markdown(f"**[{i}] {doc_name}** &nbsp;&nbsp; score: `{score:.4f}`")
            st.caption(preview)
            if i < min(2, len(sources)):
                st.divider()


# ── Render message history ────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    if message["role"] == "assistant" and message.get("sources"):
        render_sources(message["sources"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about LlamaIndex ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking ..."):
            response = st.session_state.chat_engine.chat(prompt)
            answer = str(response)
        st.markdown(answer)

    sources = getattr(response, "source_nodes", [])
    render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
