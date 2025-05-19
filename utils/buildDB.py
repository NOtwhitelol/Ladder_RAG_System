from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
import json

from dotenv import load_dotenv
import os

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

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

# Build FAISS database from metadata
def build_vector_store(metadata_list):
    docs = []
    for i, meta in enumerate(metadata_list):
        docs.append(
            Document(
                page_content=json.dumps({
                    "title": meta["title"], 
                    "summary": meta["summary"],
                    "section": meta["section"]
                })
            )
        )
    db = FAISS.from_documents(docs, embedding_model)
    return db

if __name__ == "__main__":
    json_path = "docs/metadata.json"
    md_path = "docs/Ladder_RAG_document.md"

    metadata_list = load_metadata(json_path)
    md_text = load_markdown(md_path)
    document_list = recursive_split(md_text, "###")

    embedding_model = HuggingFaceInferenceAPIEmbeddings(
        api_key=HUGGING_FACE_TOKEN,
        model_name="intfloat/multilingual-e5-large-instruct"
    )

    db = build_vector_store(metadata_list)
    db.save_local("faiss_db")
