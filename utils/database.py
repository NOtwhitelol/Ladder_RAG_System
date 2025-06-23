import os
import json
import pickle

import ollama
import faiss
import torch

from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from prompt_templates import response_templates, router_templates, rewrite_templates


# --- Configuration ---
load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
MODEL_NAME = "phi4:14b"


# --- Utilities ---
def clean_json_string(s):
    """Remove code block formatting from LLM-generated JSON"""
    s = s.strip()
    if s.startswith("```json"):
        s = s.removeprefix("```json").strip()
    if s.endswith("```"):
        s = s.removesuffix("```").strip()
    return s

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

def get_corresponding_document(indices, documents):
    """Return matched document content prefixed with headers"""
    return ["### " + documents[int(i)] for i in indices if int(i) < len(documents)]


# --- Embedding ---
class DocumentEmbedder:
    def __init__(self, model_name="intfloat/multilingual-e5-large-instruct"):
        print(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
    def embed_query(self, query):
        """Generate embedding for a query (E5 format)"""
        input_text = f"query: {query}"
        encoded_input = self.tokenizer(
            input_text, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            attention_mask = encoded_input['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            
        return embedding.cpu().numpy()


# --- Vector Store ---
class FAISSVectorStore:
    def __init__(self, index_path="faiss_db/index.faiss", documents_path="faiss_db/documents.pkl"):
        print(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        with open(documents_path, 'rb') as f:
            self.documents = pickle.load(f)
        print(f"Loaded {len(self.documents)} documents")
        self.embedder = DocumentEmbedder()
    
    def similarity_search(self, query, k=6):
        """Return top-k most similar document sections"""
        query_embedding = self.embedder.embed_query(query)
        distances, indices = self.index.search(query_embedding, k)
        
        matched_sections = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc = json.loads(self.documents[idx])
                if "section" in doc:
                    matched_sections.append(doc["section"])
        return matched_sections


# --- Load Documents ---
json_path = "docs/metadata.json"
md_path = "docs/Ladder_RAG_document.md"

metadata_list = load_metadata(json_path)
md_text = load_markdown(md_path)
document_list = recursive_split(md_text, "###")


# --- FAISS DB Instance ---
db = FAISSVectorStore()


# --- RAG Pipeline Functions ---
def search_DB(query, db_instance=db):
    """Search the vectorstore for relevant document section names"""
    return db_instance.similarity_search(query, k=6)

def search_db_sections(standalone_query):
    """Filter top 6 sections using LLM to keep the most relevant (up to 3)"""
    top_sections = search_DB(standalone_query)
    print(f"Top 6 sections: {top_sections}\n")
    source_docs = get_corresponding_document(top_sections, document_list)

    relevant_indices = []

    for idx, doc in zip(top_sections, source_docs):
        prompt = router_templates.DB_ROUTER_TEMPLATE(doc, standalone_query)
        response = ollama.chat(
            model=MODEL_NAME,
            messages=prompt,
            options={"temperature": 0.1}
        )
        answer = response['message']['content'].lower()
        
        if 'yes' in answer:
            relevant_indices.append(idx)
            if len(relevant_indices) == 3:
                break
    print(f"Related sections: {relevant_indices}")
    return relevant_indices

def run_db_rag(chat_history, original_user_query, relevant_indices):
    relevant_document_list = get_corresponding_document(relevant_indices, document_list)
    documents = "\n\n---\n\n".join(document for document in relevant_document_list)

    template = chat_history.copy()[:-1] # remove the latest user message
    template = template + response_templates.DB_RESPONSE_TEMPLATE(documents, original_user_query)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=template,
        stream=True,
        options={"temperature": 0.1}
    )

    full_response = ""

    for chunk in response:    
        message = chunk['message']['content']        
        full_response += message
        yield message

    chat_history.append({"role": "assistant", "content": full_response})
