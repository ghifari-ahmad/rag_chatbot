import os
import hashlib
import glob
from langchain_core.documents import Document
from embedding import get_embedding_function # Pastikan file ini ada
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import VectorParams, Distance

# --- IMPOR BARU UNTUK LOADING & SPLITTING ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
# ---------------------------------------------

# --- Konfigurasi ---
MARKDOWN_DATA_PATH = "markdown" # ❗️ GANTI INI: Path ke folder berisi file .md
QDRANT_PATH = "./db" # Path ke database Qdrant (misal: ./db)
COLLECTION_NAME = "doc_collection"

# =================================================================
# FUNGSI LOADING BARU (UNTUK MARKDOWN)
# =================================================================
def load_markdown_documents():
    """Memuat semua file .md dari direktori."""
    print(f"Memuat file Markdown dari: {MARKDOWN_DATA_PATH}")
    
    # Gunakan DirectoryLoader untuk memindai folder
    # Gunakan TextLoader untuk membaca konten .md
    loader = DirectoryLoader(
        MARKDOWN_DATA_PATH,
        glob="**/*.md",   # Pola untuk file markdown
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True
    )
    
    documents = loader.load()
    print(f"Total {len(documents)} file Markdown berhasil dimuat.")
    return documents

# =================================================================
# FUNGSI SPLITTING BARU (SEMANTIC CHUNKING)
# =================================================================
def split_markdown_documents(documents: list[Document]):
    """Memecah dokumen Markdown berdasarkan struktur Heading."""
    print("Memecah dokumen (Markdown) berdasarkan struktur Heading...")
    
    # Tentukan header yang akan menjadi "pembatas"
    # Sesuaikan ini jika dokumen Anda menggunakan struktur heading yang berbeda
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    # Inisialisasi splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        return_each_line=False # Kita mau blok teks utuh, bukan baris per baris
    )

    all_chunks = []
    
    # Loop untuk setiap file .md yang dimuat
    for doc in documents:
        source_file = doc.metadata.get("source", "N/A")
        
        # Panggil .split_text() pada konten Markdown
        chunks = markdown_splitter.split_text(doc.page_content)
        
        # Re-attach metadata 'source' kembali ke setiap chunk
        # dan tambahkan metadata heading (misal: "Header 3: Pasal 4")
        for chunk in chunks:
            chunk.metadata["source"] = source_file
        
        all_chunks.extend(chunks)
        
    print(f"Dokumen dipecah menjadi {len(all_chunks)} chunks semantik.")
    return all_chunks

# --- FUNGSI EMBED & STORE (SAMA SEPERTI SEBELUMNYA) ---

def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    """Membuat ID unik untuk setiap chunk."""
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        content = chunk.page_content
        # Buat ID unik berdasarkan sumber + metadata header + konten
        metadata_str = str(chunk.metadata)
        unique_str = f"{source}:{metadata_str}:{content}"
        chunk_id = hashlib.md5(unique_str.encode()).hexdigest()
        chunk.metadata["id"] = chunk_id
    return chunks

def add_to_qdrant(chunks: list[Document]):
    """Menambahkan chunks ke database Qdrant."""
    if not chunks:
        print("Tidak ada chunks untuk ditambahkan.")
        return

    client = QdrantClient(path=QDRANT_PATH)
    embedding_function = get_embedding_function()
    chunks_with_ids = calculate_chunk_ids(chunks)
    candidate_chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]

    try:
        existing_points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=candidate_chunk_ids,
            with_payload=False,
            with_vectors=False
        )
        existing_ids = {point.id for point in existing_points}
        print(f"Memeriksa {len(candidate_chunk_ids)} dokumen. Ditemukan {len(existing_ids)} yang sudah ada di DB.")
    except ValueError:
        print(f"Info: Koleksi '{COLLECTION_NAME}' belum ada. Membuat koleksi baru...")
        try:
            dummy_vec = embedding_function.embed_query("test query")
            vector_size = len(dummy_vec)
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Koleksi '{COLLECTION_NAME}' berhasil dibuat.")
            existing_ids = set()
        except Exception as emb_e:
             print(f"Error fatal saat mendapatkan ukuran vector: {emb_e}")
             return

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if len(new_chunks):
        print(f"👉 Menambahkan dokumen baru: {len(new_chunks)}")
        qdrant_db = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embedding_function # Pastikan 'embeddings' (dengan 's')
        )
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        qdrant_db.add_documents(new_chunks, ids=new_chunk_ids)
        print(f"✅ Berhasil menambahkan {len(new_chunks)} dokumen baru.")
    else:
        print("✅ Database sudah ter-update (tidak ada dokumen baru).")

# --- MAIN EKSEKUSI ---
if __name__ == "__main__":
    # 1. Hapus DB lama (opsional, tapi disarankan jika struktur berubah)
    # Ganti QDRANT_PATH jika berbeda
    if os.path.exists("./db"):
        print("Menghapus database Qdrant lama...")
        import shutil
        shutil.rmtree("./db")
    
    # 2. Buat direktori baru
    os.makedirs(QDRANT_PATH, exist_ok=True)
    
    # 3. Jalankan pipeline
    documents = load_markdown_documents()
    chunks = split_markdown_documents(documents)
    add_to_qdrant(chunks)
    
    print("\n--- Proses Ingest Selesai ---")
    print(f"Database Qdrant Anda sekarang ada di: {QDRANT_PATH}")