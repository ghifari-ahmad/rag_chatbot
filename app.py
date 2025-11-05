import streamlit as st
import os
import hashlib
import glob
from langchain_core.documents import Document
from embedding import get_embedding_function
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import VectorParams, Distance
# === PERUBAHAN ===
# Menghapus flashrank, menambahkan FlagEmbedding
from FlagEmbedding import FlagReranker 
# =================
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI # Client untuk llama.cpp
import nest_asyncio

# Terapkan patch asyncio (diperlukan jika ada sisa logika async)
nest_asyncio.apply()

# --- Konfigurasi ---
QDRANT_PATH = "./db" # Sesuaikan dengan path ingest
COLLECTION_NAME = "doc_collection"
# FLASHRANK_CACHE_DIR = "./flashrank_cache" # <-- Tidak diperlukan lagi
DATA_PATH = "./markdown"

# --- Konfigurasi Server Llama.cpp ---
LLAMA_CPP_SERVER_URL = "http://localhost:8080/v1" 
LLAMA_CPP_MODEL_NAME = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# ==========================================================
# --- PERBAIKAN PROMPT TEMPLATE ---
# Gunakan format [INST] asli dari Mistral Instruct
# ==========================================================
PROMPT_TEMPLATE = """<s>[INST]
Gunakan potongan informasi berikut untuk menjawab pertanyaan pengguna.
Jika Anda tidak tahu jawabannya, katakan saja bahwa Anda tidak tahu.

Konteks:
{context}

---
Jawablah pertanyaan dalam bahasa Indonesia.
Pertanyaan: {question}
[/INST]"""
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
        embedding=embedding_function # <-- Bug 'embedding=' DIPERBAIKI
    )

    # 2. Muat Ranker
    # ==========================================================
    # --- PERUBAHAN DI SINI ---
    # Mengganti flashrank dengan FlagReranker
    # ==========================================================
    print("--- ⚠️ Memuat FlagReranker ---")
    ranker = FlagReranker(
        'BAAI/bge-reranker-v2-m3', 
        use_fp16=True # Gunakan FP16 untuk performa lebih cepat
    )
    print("--- ✅ FlagReranker berhasil dimuat ---")
    # ==========================================================

    # 3. Hubungkan ke Llama Server (API)
    model = ChatOpenAI(
        openai_api_key="not-needed",
        openai_api_base=LLAMA_CPP_SERVER_URL,
        model=LLAMA_CPP_MODEL_NAME,
        temperature=0.1,
        max_tokens=1024
    )

    # 4. Buat RAG Chain dengan prompt yang benar
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    parser = StrOutputParser()
    chain = prompt_template | model | parser

    print("--- ✅ Komponen dan RAG chain berhasil dimuat ---")
    return qdrant_db, ranker, chain

def get_rag_response(query_text, db, ranker, chain):
    """Menjalankan pipeline RAG (Retrieve, Rerank, Generate)."""
    
    # 1. Retrieve
    try:
        retrieved_docs = db.similarity_search(query_text, k=20) # Ambil 20 untuk reranker
    except Exception as e:
        return f"Error: Gagal mengambil data dari Qdrant ({e}). Pastikan database '{QDRANT_PATH}' sudah dibuat oleh 'ingest.py'.", []
        
    if not retrieved_docs:
        return "Maaf, saya tidak menemukan dokumen yang relevan.", []

    # ==========================================================
    # --- PERUBAHAN DI SINI ---
    # Logika reranking diubah total untuk FlagReranker
    # ==========================================================
    
    # 2. Rerank
    # Buat pasangan [query, passage] untuk reranker
    sentence_pairs = [[query_text, doc.page_content] for doc in retrieved_docs]

    # Hitung skor
    scores = ranker.compute_score(sentence_pairs)

    # Gabungkan dokumen asli dengan skornya
    doc_with_scores = list(zip(retrieved_docs, scores))
    
    # Urutkan berdasarkan skor (dari tertinggi ke terendah)
    sorted_docs_with_scores = sorted(doc_with_scores, key=lambda x: x[1], reverse=True)

    # Ambil 3 chunk teratas untuk konteks
    top_k = 3
    # Ambil hanya objek Document-nya saja
    top_reranked_docs = [doc for doc, score in sorted_docs_with_scores[:top_k]]

    if not top_reranked_docs:
         return "Tidak ditemukan hasil yang relevan setelah proses reranking.", []
    # ==========================================================

    # 3. Siapkan Konteks dan Sumber
    sources = []
    context_text = ""
    
    # === PERUBAHAN ===
    # Kita sekarang me-loop 'top_reranked_docs' (list of Document)
    # Bukan 'reranked_results' (list of dict)
    # =================
    for doc in top_reranked_docs:
        context_text += doc.page_content + "\n\n---\n\n"
        meta = doc.metadata # <-- Akses metadata langsung
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

st.set_page_config(page_title="Chat RAG GGUF", page_icon="🤖")
st.title("🤖 Chatbot RAG (llama.cpp Server)")

# --- Sidebar (Sekarang hanya untuk info) ---
with st.sidebar:
    st.header("⚙️ Konfigurasi Sistem")
    st.write(f"Folder data: `{DATA_PATH}`")
    st.write(f"Database: `{QDRANT_PATH}`")
    st.caption(f"LLM Server: `{LLAMA_CPP_SERVER_URL}`")
    st.divider()
    st.info("Database harus dibuat/diperbarui dengan menjalankan 'ingest_md.py' secara terpisah.")
    if st.button("Hapus Cache & Muat Ulang Model"):
        st.cache_resource.clear()
        # st.experimental_rerun() # <-- 'experimental_rerun' sudah usang
        st.rerun() # <-- Gunakan st.rerun()


# --- Inisialisasi Chat ---
try:
    db, ranker, chain = load_all_components() 
except Exception as e:
    st.error(f"Gagal memuat komponen RAG: {e}")
    st.error(f"Pastikan server llama.cpp Anda sudah berjalan di {LLAMA_CPP_SERVER_URL}")
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
        with st.spinner("Sedang berpikir... 🧠 (Menghubungi server llama.cpp...)"):
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
                st.error("Pastikan server llama.cpp Anda masih berjalan.")