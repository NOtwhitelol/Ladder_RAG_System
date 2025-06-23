import os
import json
import pickle

import faiss
import torch

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
os.environ["HUGGINGFACE_TOKEN"] = HUGGING_FACE_TOKEN


# --- File Utilities ---
def load_metadata(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_markdown(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        return f.read()

def recursive_split(text, delimiter):
    if delimiter not in text:
        return [text]
    
    split_index = text.index(delimiter)
    before_delimiter = text[:split_index].strip()
    after_delimiter = text[split_index + len(delimiter):].strip()
    
    return [before_delimiter] + recursive_split(after_delimiter, delimiter)


# --- Embedding Model ---
class DocumentEmbedder:
    def __init__(self, model_name="intfloat/multilingual-e5-large-instruct"):
        print(f"[Embedder] Initializing model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Embedder] Using device: {self.device}")
        self.model.to(self.device)
        
    def embed_documents(self, texts):
        """Generate embeddings for a batch of texts."""
        inputs = ["query: " + text for text in texts] # E5 format
        
        encoded_inputs = self.tokenizer(
            inputs, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            attention_mask = encoded_inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
        return embeddings.cpu().numpy()

# --- FAISS Index ---
class FAISSDatabase:
    def __init__(self, embedding_dim=1024):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
    
    def add_documents(self, documents, embeddings):
        """Add documents and corresponding embeddings to FAISS index."""
        self.documents.extend(documents)
        self.index.add(embeddings)
    
    def save(self, index_path, documents_path):
        """Save FAISS index and document list to disk."""
        faiss.write_index(self.index, index_path)
        with open(documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
    
    def search(self, query_embedding, k=5):
        """Search for the top-k most similar documents."""
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'distance': float(distances[0][i])
                })
        return results

# --- Vector Store Builder ---
def build_vector_store(metadata_list, embedder):
    print(f"[VectorStore] Embedding {len(metadata_list)} documents...")
    documents = []
    
    for meta in metadata_list:
        doc_content = json.dumps({
            "title": meta["title"], 
            "summary": meta["summary"],
            "section": meta["section"]
        })
        documents.append(doc_content)
    
    embeddings = embedder.embed_documents(documents)
    db = FAISSDatabase(embedding_dim=embeddings.shape[1])
    db.add_documents(documents, embeddings)
    return db, documents

if __name__ == "__main__":
    print("[System] Starting vector store creation...")
    json_path = "docs/metadata.json"
    md_path = "docs/Ladder_RAG_document.md"

    metadata_list = load_metadata(json_path)
    md_text = load_markdown(md_path)
    document_list = recursive_split(md_text, "###")
    
    print(f"[System] Loaded {len(metadata_list)} metadata items")
    
    embedder = DocumentEmbedder(model_name="intfloat/multilingual-e5-large-instruct")
    db, documents = build_vector_store(metadata_list, embedder)
    
    os.makedirs("faiss_db", exist_ok=True)
    print("[System] Saving FAISS index and documents...")
    db.save("faiss_db/index.faiss", "faiss_db/documents.pkl")

    print(f"[System] Completed. Stored {len(documents)} documents.")