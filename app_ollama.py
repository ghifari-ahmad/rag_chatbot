import streamlit as st
import os
import hashlib
import glob
from langchain_core.documents import Document
from embedding import get_embedding_function
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import VectorParams, Distance
from FlagEmbedding import FlagReranker 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# === PERUBAHAN ===
# Mengganti ChatOpenAI (untuk llama.cpp) dengan ChatOllama
from langchain_community.chat_models import ChatOllama
# =================
import nest_asyncio

# Terapkan patch asyncio
nest_asyncio.apply()

# --- Konfigurasi ---
QDRANT_PATH = "./db" 
COLLECTION_NAME = "doc_collection"
DATA_PATH = "./markdown"

# === PERUBAHAN ===
# --- Konfigurasi Server Ollama ---
# Server llama.cpp tidak lagi digunakan
OLLAMA_SERVER_URL = "http://localhost:11434" 
OLLAMA_MODEL_NAME = "mistral" # Ganti ini jika nama model Anda di Ollama berbeda
# ==========================================================


# ==========================================================
# --- PERBAIKAN PROMPT TEMPLATE ---
# ChatOllama akan menangani format [INST] Mistral secara otomatis.
# Kita sediakan System prompt dan Human prompt secara terpisah
# untuk hasil terbaik dengan LangChain.
# ==========================================================
SYSTEM_PROMPT = """Gunakan potongan informasi berikut untuk menjawab pertanyaan pengguna.
Jika Anda tidak tahu jawabannya, katakan saja bahwa Anda tidak tahu.

Konteks:
{context}

---
Jawablah pertanyaan dalam bahasa Indonesia.
Pertanyaan: {question}"""
# ==========================================================


# --- FUNGSI RAG (HANYA QUERY) ---

@st.cache_resource
def load_all_components():
    """Memuat DB, Ranker, dan RAG Chain."""
    print("--- ⚠️ MEMUAT KOMPONEN (DB, RANKER, LLM CHAIN) ---")

    # 1. Muat Qdrant DB
    client = QdrantClient(path=QDRANT_PATH)
    embedding_function = get_embedding_function()
    qdrant_db = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_function
    )

    # 2. Muat Ranker
    print("--- ⚠️ Memuat FlagReranker ---")
    ranker = FlagReranker(
        'BAAI/bge-reranker-v2-m3', 
        use_fp16=True 
    )
    print("--- ✅ FlagReranker berhasil dimuat ---")

    # === PERUBAHAN ===
    # 3. Hubungkan ke Ollama Server
    model = ChatOllama(
        base_url=OLLAMA_SERVER_URL,
        model=OLLAMA_MODEL_NAME,
        temperature=0.1
    )
    # =================

    # === PERUBAHAN ===
    # 4. Buat RAG Chain dengan prompt yang benar (System + Human)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    # =================
    
    parser = StrOutputParser()
    chain = prompt_template | model | parser

    print("--- ✅ Komponen dan RAG chain berhasil dimuat ---")
    return qdrant_db, ranker, chain

def get_rag_response(query_text, db, ranker, chain):
    """Menjalankan pipeline RAG (Retrieve, Rerank, Generate)."""
    
    # 1. Retrieve
    try:
        retrieved_docs = db.similarity_search(query_text, k=20) 
    except Exception as e:
        return f"Error: Gagal mengambil data dari Qdrant ({e}). Pastikan database '{QDRANT_PATH}' sudah dibuat oleh 'ingest.py'.", []
        
    if not retrieved_docs:
        return "Maaf, saya tidak menemukan dokumen yang relevan.", []

    # 2. Rerank (Logika FlagReranker tetap sama)
    sentence_pairs = [[query_text, doc.page_content] for doc in retrieved_docs]
    scores = ranker.compute_score(sentence_pairs)
    doc_with_scores = list(zip(retrieved_docs, scores))
    sorted_docs_with_scores = sorted(doc_with_scores, key=lambda x: x[1], reverse=True)
    
    top_k = 3
    top_reranked_docs = [doc for doc, score in sorted_docs_with_scores[:top_k]]

    if not top_reranked_docs:
            return "Tidak ditemukan hasil yang relevan setelah proses reranking.", []

    # 3. Siapkan Konteks dan Sumber (Logika tetap sama)
    sources = []
    context_text = ""
    
    for doc in top_reranked_docs:
        context_text += doc.page_content + "\n\n---\n\n"
        meta = doc.metadata 
        source_file = os.path.basename(meta.get("source", "N/A"))
        sources.append(source_file)
    
    unique_sources = sorted(list(set(sources)))

    # 4. Generate respon
    response_text = chain.invoke({
        "context": context_text,
        "question": query_text
    })
    
    return response_text, unique_sources

# --------------------------------------------
# --- INTERFACE STREAMLIT ---
# --------------------------------------------

st.set_page_config(page_title="Chat RAG Ollama", page_icon="🤖")
st.title("🤖 Chatbot RAG (Ollama Server)") # === JUDUL DIGANTI ===

# --- Sidebar (Info disesuaikan untuk Ollama) ---
with st.sidebar:
    st.header("⚙️ Konfigurasi Sistem")
    st.write(f"Folder data: `{DATA_PATH}`")
    st.write(f"Database: `{QDRANT_PATH}`")
    # === PERUBAHAN ===
    st.caption(f"LLM Server (Ollama): `{OLLAMA_SERVER_URL}`")
    st.caption(f"Model: `{OLLAMA_MODEL_NAME}`")
    # =================
    st.divider()
    st.info("Database harus dibuat/diperbarui dengan menjalankan 'ingest_md.py' secara terpisah.")
    if st.button("Hapus Cache & Muat Ulang Model"):
        st.cache_resource.clear()
        st.rerun() 


# --- Inisialisasi Chat ---
try:
    db, ranker, chain = load_all_components() 
except Exception as e:
    st.error(f"Gagal memuat komponen RAG: {e}")
    # === PERUBAHAN ===
    st.error(f"Pastikan server Ollama Anda sudah berjalan di {OLLAMA_SERVER_URL} dan model '{OLLAMA_MODEL_NAME}' sudah tersedia (misal: `ollama pull mistral`).")
    # =================
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya chatbot RAG Anda. Silakan tanya sesuatu dari dokumen Anda."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tulis pertanyaan Anda..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # === PERUBAHAN ===
        with st.spinner("Sedang berpikir... 🧠 (Menghubungi server Ollama...)"):
        # =================
            try:
                response_text, sources = get_rag_response(prompt, db, ranker, chain)
                
                if sources:
                    source_list = "\n- ".join(sorted(list(set(sources))))
                    full_response = f"{response_text}\n\n---\n*Sumber:*\n- {source_list}"
                else:
                    full_response = response_text
                    
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Gagal mendapat respons dari LLM: {e}")
                # === PERUBAHAN ===
                st.error("Pastikan server Ollama Anda masih berjalan.")
                # =================