from langchain_community.embeddings.ollama import OllamaEmbeddings
# Impor HuggingFaceBgeEmbeddings, bukan FastEmbed
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import torch

def get_embedding_function():
    """
    Menggunakan HuggingFaceBgeEmbeddings untuk memuat bge-m3.
    Ini menggunakan library 'sentence-transformers'.
    """
        
    # Tentukan device. 'cuda' jika ada GPU, jika tidak 'cpu'
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": device}
        
    # 'normalize_embeddings=True' sangat disarankan untuk model BGE
    encode_kwargs = {"normalize_embeddings": True} 

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
        # 'query_instruction' secara otomatis ditangani oleh HuggingFaceBgeEmbeddings
    )
        
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
    
