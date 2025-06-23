import json

import ollama
from prompt_templates import response_templates, router_templates, rewrite_templates

MODEL_NAME = "phi4:14b"

def clean_json_string(s):
    """Remove code block formatting from LLM-generated JSON"""
    s = s.strip()
    if s.startswith("```json"):
        s = s.removeprefix("```json").strip()
    if s.endswith("```"):
        s = s.removesuffix("```").strip()
    return s

def generate_standalone_query(chat_history, query):
    response = ollama.chat(
        model=MODEL_NAME,
        messages=rewrite_templates.STANDALONE_QUESTION_REWRITE_TEMPLATE(chat_history, query),
        options={"temperature": 0}
    )

    rewrite_query = clean_json_string(response['message']['content'])
    print(f"Rewrite Query: {rewrite_query}")

    try:
        data = json.loads(rewrite_query)
        return data["question"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return rewrite_query  # Fallback: return as plain string

def is_close_domain_query(standalone_query) -> bool:
    template = router_templates.ROUTER_TEMPLATE(standalone_query)
    
    response = ollama.chat(
        model=MODEL_NAME,
        messages=template,
        options={"temperature": 0.1}
    )
    answer = response['message']['content']
    # print(f"Answer: {answer}")
    
    if "web search" in answer:
        return False
    else:
        return True

def direct_response(chat_history, original_user_query):
    template = chat_history.copy()[:-1] # remove the latest user message
    template.append({"role": "user", "content": original_user_query})
    
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