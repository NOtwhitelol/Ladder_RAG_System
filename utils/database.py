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

# 讀取 Metadata JSON 檔案
def load_metadata(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 讀取 Markdown 文件
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

# 建立 FAISS 向量資料庫（索引 metadata）
def build_vector_store(metadata_list):
    docs = [
        Document(page_content=json.dumps({"title": meta["title"], "summary": meta["summary"]}))
        for meta in metadata_list
    ]
    db = FAISS.from_documents(docs, embedding_model)
    return db

def get_corresponding_strings(indices, strings):
    return "\n\n---\n\n".join("### " + strings[int(i)] for i in indices if int(i) < len(strings))

json_path = "docs/metadata.json"
md_path = "docs/Ladder_RAG_document.md"

metadata_list = load_metadata(json_path)
md_text = load_markdown(md_path)
split_result = recursive_split(md_text, "###")

embedding_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGING_FACE_TOKEN,
    model_name="intfloat/multilingual-e5-small"
)


db = build_vector_store(metadata_list)

def search_DB(query, db, metadata_list, split_result):
    # 在 metadata 上進行相似度檢索
    result_simi = db.similarity_search(query, k=3)

    # 匹配 metadata，找到對應的原始內容
    matched_sections = []
    for result in result_simi:
        metadata = json.loads(result.page_content)
        for meta in metadata_list:
            if meta["summary"] == metadata["summary"]:
                matched_sections.append(meta["section"])  # section_link 存的是原始內容
                break
    print(f"matched_sections: {matched_sections}")

    # 輸出最終結果
    source_knowledge = get_corresponding_strings(matched_sections, split_result)
    # print(f"Source knowledge: \n{source_knowledge}\n")
    return source_knowledge

def DB_search_router(query):
    source_knowledge = search_DB(query, db, metadata_list, split_result)
    print(f"Source knowledge: \n{source_knowledge}\n")
    
    template = router_templates.DB_ROUTER_TEMPLATE(source_knowledge, query)
    
    response = ollama.chat(
        model="ladder_llama3.1",
        messages=template
    )
    answer = response['message']['content'].lower()
    
    try: 
        if 'yes' in answer:
            return True
        elif 'no' in answer:
            return False
    
    except Exception as e:
        print(f"Error occurred in DB search router: {e}")
        return DB_search_router(query)

def Run_DB_RAG(chat_history, follow_up_question, standalone_query):
    result_simi = db.similarity_search(standalone_query , k = 3)
    source_knowledge = "\n\n".join([x.page_content for x in result_simi])

    template = chat_history.copy()
    template = template + response_templates.DB_RESPONSE_TEMPLATE(source_knowledge, follow_up_question)

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
