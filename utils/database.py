from prompt_templates import response_templates, router_templates, rewrite_templates
import ollama
import json
import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

from dotenv import load_dotenv
import os
import re

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
MODEL_NAME = "phi4:14b"

def clean_json_string(s):
    s = s.strip()
    if s.startswith("```json"):
        s = s.removeprefix("```json").strip()
    if s.endswith("```"):
        s = s.removesuffix("```").strip()
    return s

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

def get_corresponding_document(indices, documents):
    return ["### " + documents[int(i)] for i in indices if int(i) < len(documents)]

class DocumentEmbedder:
    def __init__(self, model_name="intfloat/multilingual-e5-large-instruct"):
        print(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
    def embed_query(self, query):
        """Generate embedding for a query"""
        # E5 models expect "query: " prefix for queries
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
            # Mean pooling
            attention_mask = encoded_input['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            
        return embedding.cpu().numpy()

class FAISSVectorStore:
    def __init__(self, index_path="faiss_db/index.faiss", documents_path="faiss_db/documents.pkl"):
        """Load the FAISS index and documents"""
        print(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        with open(documents_path, 'rb') as f:
            self.documents = pickle.load(f)
        print(f"Loaded {len(self.documents)} documents")
        
        # Initialize embedder
        self.embedder = DocumentEmbedder()
    
    def similarity_search(self, query, k=6):
        """Search for similar documents and return sections"""
        # Get query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Search in the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Get matching sections
        matched_sections = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc = json.loads(self.documents[idx])
                if "section" in doc:
                    matched_sections.append(doc["section"])
                    
        return matched_sections

json_path = "docs/metadata.json"
md_path = "docs/Ladder_RAG_document.md"

metadata_list = load_metadata(json_path)
md_text = load_markdown(md_path)
document_list = recursive_split(md_text, "###")

db = FAISSVectorStore()

def search_DB(query, db=db):
    """Search the database for similar documents"""
    return db.similarity_search(query, k=6)

def DB_search_router(standalone_query):
    section_indices = search_DB(standalone_query, db)
    print(f"Top 6 sections: {section_indices}\n")

    source_documents = get_corresponding_document(section_indices, document_list)

    related_section_indices = []

    for index, document in zip(section_indices, source_documents):
        template = router_templates.DB_ROUTER_TEMPLATE(document, standalone_query)
        response = ollama.chat(
            model=MODEL_NAME,
            messages=template,
            options={"temperature": 0.1}
        )
        answer = response['message']['content'].lower()
        
        if 'yes' in answer:
            related_section_indices.append(index)
            if len(related_section_indices) == 3:
                break
    print(f"Related sections: {related_section_indices}")
    return related_section_indices

def Run_DB_RAG(chat_history, original_user_query, related_section_indices):
    related_document_list = get_corresponding_document(related_section_indices, document_list)
    documents = "\n\n---\n\n".join(document for document in related_document_list)
    print(f"Related Documents: \n{documents}\n")

    template = chat_history.copy()[:-1] # remove the latest user message
    template = template + response_templates.DB_RESPONSE_TEMPLATE(documents, original_user_query)

    # print(template)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=template,
        stream=True,
        options={"temperature": 0.1}
    )

    full_response = ""
    response_started = False
    thinking_section = ""

# When using a normal LLM
    for chunk in response:    
        message = chunk['message']['content']        
        full_response += message
        yield message
# When using an LLM with think mode
    # for chunk in response:
    #     message = chunk['message']['content']
    #     if not response_started:
    #         thinking_section += message
    #         end_tag = "</think>"
    #         if end_tag in thinking_section:
    #             # Strip out the <think>...</think> block
    #             think_match = re.search(r"<think>(.*?)</think>", thinking_section, flags=re.DOTALL)
    #             if think_match:
    #                 print("Thinking section:", think_match.group(1).strip())
    #             response_started = True
    #     else:
    #         full_response += message
    #         yield message

    chat_history.append({"role": "assistant", "content": full_response})
