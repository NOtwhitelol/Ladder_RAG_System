import ollama
from prompt_templates import response_templates, router_templates, rewrite_templates
import json
import re

MODEL_NAME = "phi4:14b"

def clean_json_string(s):
    s = s.strip()
    if s.startswith("```json"):
        s = s.removeprefix("```json").strip()
    if s.endswith("```"):
        s = s.removesuffix("```").strip()
    return s

def standalone_query_rewrite(chat_history, query):
    response = ollama.chat(
        model=MODEL_NAME,
        messages=rewrite_templates.STANDALONE_QUESTION_REWRITE_TEMPLATE(chat_history, query),
        options={"temperature": 0}
    )
    # rewrite_query = re.sub(r"<think>.*?</think>", "", response['message']['content'], flags=re.DOTALL)
    rewrite_query = clean_json_string(response['message']['content'])
    print(f"Rewrite Query: {rewrite_query}")

    try:
        data = json.loads(rewrite_query)
        return data["question"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error parsing rewritten query: {e}")
        print(f"Return the rewrite query without JSONify: {rewrite_query}")
        return rewrite_query  # Return the rewrite query without JSONify

def Router(standalone_query):
    template = router_templates.ROUTER_TEMPLATE(standalone_query)
    
    response = ollama.chat(
        model=MODEL_NAME,
        messages=template,
        options={"temperature": 0.1}
    )
    answer = response['message']['content']
    # print(f"Answer: {answer}")
    
    try:
        if 'vectorstore' in answer:
            return True
        elif 'web search' in answer:
            return False
    
    except Exception as e:
        print(f"Error occurred in router: {e}")
        return Router(standalone_query)


def Run_Direct_RAG(chat_history, original_user_query):
    template = chat_history.copy()[:-1] # remove the latest user message
    template.append({"role": "user", "content": original_user_query})
    
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
    