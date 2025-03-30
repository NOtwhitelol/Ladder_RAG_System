from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from prompt_templates import response_templates, router_templates, rewrite_templates
import ollama
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

def get_corresponding_document(indices, documents):
    return ["### " + documents[int(i)] for i in indices if int(i) < len(documents)]


json_path = "docs/metadata.json"
md_path = "docs/Ladder_RAG_document.md"

metadata_list = load_metadata(json_path)
md_text = load_markdown(md_path)
document_list = recursive_split(md_text, "###")

embedding_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGING_FACE_TOKEN,
    model_name="intfloat/multilingual-e5-small"
)


db = build_vector_store(metadata_list)

def search_DB(query, db):
    search_results = db.similarity_search(query, k=6)

    matched_sections = []
    for result in search_results:
        result_data = json.loads(result.page_content)
        if "section" in result_data:
            matched_sections.append(result_data["section"])
            
    return matched_sections

def DB_search_router(query):
    section_indices = search_DB(query, db)
    print(f"Top 6 sections: {section_indices}\n")

    source_documents = get_corresponding_document(section_indices, document_list)

    related_section_indices = []

    for index, document in zip(section_indices, source_documents):
        template = router_templates.DB_ROUTER_TEMPLATE(document, query)
        response = ollama.chat(
            model="ladder_llama3.1",
            messages=template
        )
        answer = response['message']['content'].lower()
        
        if 'yes' in answer:
            related_section_indices.append(index)
            if len(related_section_indices) == 3:
                break
    print(f"Related sections: {related_section_indices}")
    return related_section_indices

def Run_DB_RAG(chat_history, follow_up_question, related_section_indices):
    related_document_list = get_corresponding_document(related_section_indices, document_list)
    documents = "\n\n---\n\n".join("### " + document for document in related_document_list)
    print(f"Related Documents: \n{documents}\n")

    template = chat_history.copy()
    template = template + response_templates.DB_RESPONSE_TEMPLATE(documents, follow_up_question)

    response = ollama.chat(
        model="ladder_llama3.1",
        messages=template,
        stream=True
    )

    full_response = ""

    for chunk in response:
        message = chunk['message']['content']
        full_response += message
        yield message

    chat_history.append({"role": "assistant", "content": full_response})
