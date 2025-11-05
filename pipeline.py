import os
import sys  # Diperlukan untuk error handling dan exit
from embedding import get_embedding_function
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from FlagEmbedding import FlagReranker
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
import nest_asyncio

# Terapkan patch asyncio
nest_asyncio.apply()

# --- Konfigurasi (Tetap sama) ---
QDRANT_PATH = "./db"
COLLECTION_NAME = "doc_collection"
DATA_PATH = "./markdown" # (Informasional)

# --- Konfigurasi Server Ollama (Tetap sama) ---
OLLAMA_SERVER_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "mistral"

# --- PERBAIKAN PROMPT TEMPLATE (Tetap sama) ---
SYSTEM_PROMPT = """Gunakan potongan informasi berikut untuk menjawab pertanyaan pengguna.
Jika Anda tidak tahu jawabannya, katakan saja bahwa Anda tidak tahu.

Konteks:
{context}

---
Jawablah pertanyaan dalam bahasa Indonesia.
Pertanyaan: {question}"""


# --- FUNGSI RAG (HANYA QUERY) ---

# @st.cache_resource Dihapus
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

    # 3. Hubungkan ke Ollama Server
    model = ChatOllama(
        base_url=OLLAMA_SERVER_URL,
        model=OLLAMA_MODEL_NAME,
        temperature=0.1
    )

    # 4. Buat RAG Chain
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

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
        return f"Error: Gagal mengambil data dari Qdrant ({e}). Pastikan database '{QDRANT_PATH}' sudah dibuat.", []
        
    if not retrieved_docs:
        return "Maaf, saya tidak menemukan dokumen yang relevan.", []

    # 2. Rerank
    sentence_pairs = [[query_text, doc.page_content] for doc in retrieved_docs]
    scores = ranker.compute_score(sentence_pairs)
    doc_with_scores = list(zip(retrieved_docs, scores))
    sorted_docs_with_scores = sorted(doc_with_scores, key=lambda x: x[1], reverse=True)
    
    top_k = 3
    top_reranked_docs = [doc for doc, score in sorted_docs_with_scores[:top_k]]

    if not top_reranked_docs:
            return "Tidak ditemukan hasil yang relevan setelah proses reranking.", []

    # 3. Siapkan Konteks dan Sumber
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
# --- INTERFACE TERMINAL (PENGGANTI STREAMLIT) ---
# --------------------------------------------

def main():
    """Fungsi utama untuk menjalankan loop chat di terminal."""
    
    # --- Inisialisasi ---
    try:
        db, ranker, chain = load_all_components()
    except Exception as e:
        print(f"\n[ERROR FATAL] Gagal memuat komponen RAG: {e}", file=sys.stderr)
        print(f"Pastikan server Ollama Anda sudah berjalan di {OLLAMA_SERVER_URL}", file=sys.stderr)
        print(f"dan model '{OLLAMA_MODEL_NAME}' sudah tersedia (misal: `ollama pull mistral`).", file=sys.stderr)
        print(f"Pastikan juga database Qdrant ada di: {QDRANT_PATH}", file=sys.stderr)
        sys.exit(1) # Keluar dari program jika komponen gagal dimuat

    print("\n" + "=" * 50)
    print("🤖 Chatbot RAG (Ollama Server) - Versi Terminal")
    print(f"Model: {OLLAMA_MODEL_NAME} | DB: {QDRANT_PATH}")
    print("Ketik 'exit' atau 'keluar' untuk berhenti.")
    print("=" * 50 + "\n")
    print("Halo! Saya chatbot RAG Anda. Silakan tanya sesuatu dari dokumen Anda.")

    # --- Loop Chat ---
    while True:
        try:
            # Dapatkan input pengguna
            prompt = input("Anda: ")
            
            if not prompt:
                continue
                
            # Cek perintah keluar
            prompt_lower = prompt.lower()
            if prompt_lower == 'exit' or prompt_lower == 'keluar':
                print("\nAsisten: Sampai jumpa!")
                break
            
            # Tampilkan pesan "berpikir"
            print("\nAsisten: Sedang berpikir... 🧠 (Menghubungi server Ollama...)")
            
            # Dapatkan respons RAG
            response_text, sources = get_rag_response(prompt, db, ranker, chain)
            
            # Format dan cetak respons
            if sources:
                source_list = "\n- ".join(sorted(list(set(sources))))
                full_response = f"{response_text}\n\n---\n*Sumber:*\n- {source_list}"
            else:
                full_response = response_text
                
            print(full_response)
            print("-" * 50)

        except KeyboardInterrupt:
            # Handle jika pengguna menekan Ctrl+C
            print("\nAsisten: Sesi dihentikan. Sampai jumpa!")
            break
        except Exception as e:
            # Handle error runtime
            print(f"\n[ERROR] Terjadi kesalahan: {e}", file=sys.stderr)
            print("Silakan coba pertanyaan lain.")
            print("-" * 50)

if __name__ == "__main__":
    main()