import json
import os
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
import pickle

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# Set Hugging Face token for API access
os.environ["HUGGINGFACE_TOKEN"] = HUGGING_FACE_TOKEN

def load_metadata(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_markdown(md_path):
    with open(md_path, 'r', encoding='utf-8') as file:
        return file.read()

def recursive_split(text, delimiter):
    if delimiter not in text:
        return [text]
    
    split_index = text.index(delimiter)
    before_delimiter = text[:split_index].strip()
    after_delimiter = text[split_index + len(delimiter):].strip()
    
    return [before_delimiter] + recursive_split(after_delimiter, delimiter)

class DocumentEmbedder:
    def __init__(self, model_name="intfloat/multilingual-e5-large-instruct"):
        print(f"Initializing embedder with model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
    def embed_documents(self, texts):
        """Generate embeddings for a batch of texts"""
        # Prepare instruction-based inputs (as per E5 model requirements)
        inputs = ["query: " + text for text in texts]
        
        # Tokenize inputs
        encoded_inputs = self.tokenizer(
            inputs, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        # Get model embeddings
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            # Mean pooling - use attention mask to handle padding
            attention_mask = encoded_inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
        # Convert to numpy for FAISS
        return embeddings.cpu().numpy()

class FAISSDatabase:
    def __init__(self, embedding_dim=1024):
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (Euclidean)
        self.documents = []
    
    def add_documents(self, documents, embeddings):
        """Add documents and their embeddings to the index"""
        self.documents.extend(documents)
        self.index.add(embeddings)
    
    def save(self, index_path, documents_path):
        """Save the FAISS index and documents"""
        faiss.write_index(self.index, index_path)
        with open(documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
    
    def search(self, query_embedding, k=5):
        """Search for similar documents"""
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):  # -1 means no results
                results.append({
                    'document': self.documents[idx],
                    'distance': float(distances[0][i])
                })
        return results

def build_vector_store(metadata_list, embedder):
    print(f"Building vector store for {len(metadata_list)} documents")
    documents = []
    
    # Prepare documents for embedding
    for meta in metadata_list:
        doc_content = json.dumps({
            "title": meta["title"], 
            "summary": meta["summary"],
            "section": meta["section"]
        })
        documents.append(doc_content)
    
    # Get embeddings
    embeddings = embedder.embed_documents(documents)
    
    # Create and populate FAISS index
    db = FAISSDatabase(embedding_dim=embeddings.shape[1])
    db.add_documents(documents, embeddings)
    
    return db, documents

if __name__ == "__main__":
    print("Starting vector store creation...")
    json_path = "docs/metadata.json"
    md_path = "docs/Ladder_RAG_document.md"

    metadata_list = load_metadata(json_path)
    md_text = load_markdown(md_path)
    document_list = recursive_split(md_text, "###")
    
    print(f"Loaded {len(metadata_list)} metadata items")
    
    # Create embedder
    embedder = DocumentEmbedder(model_name="intfloat/multilingual-e5-large-instruct")
    
    # Build vector store
    db, documents = build_vector_store(metadata_list, embedder)
    
    # Save to disk
    os.makedirs("faiss_db", exist_ok=True)
    print("Saving FAISS index and documents...")
    db.save("faiss_db/index.faiss", "faiss_db/documents.pkl")
    print(f"FAISS database saved with {len(documents)} documents")