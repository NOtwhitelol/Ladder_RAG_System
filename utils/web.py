from tavily import TavilyClient
from prompt_templates import response_templates, router_templates, rewrite_templates
import ollama
import json

from dotenv import load_dotenv
import os
import re

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

client = TavilyClient(api_key=TAVILY_API_KEY)
MODEL_NAME = "phi4:14b"

def clean_json_string(s):
    s = s.strip()
    if s.startswith("```json"):
        s = s.removeprefix("```json").strip()
    if s.endswith("```"):
        s = s.removesuffix("```").strip()
    return s


def Web_query_rewrite(standalone_query):
    template = rewrite_templates.WEB_QUERY_REWRITE_TEMPLATE(standalone_query)
    response = ollama.chat(
        model=MODEL_NAME,
        messages=template,
        options={"temperature": 0.1}
    )
    web_rewrite_query = clean_json_string(response['message']['content'])
    print(f"Web Rewrite Query: {web_rewrite_query}")
    
    try:
        answer = json.loads(web_rewrite_query)
        return answer["query"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing rewritten query: {e}")
        print(f"Return the rewrite query without JSONify: {web_rewrite_query}")
        return web_rewrite_query  # Return the rewrite query without JSONify


def Web_search_router(standalone_query):
    template = router_templates.WEB_ROUTER_TEMPLATE(standalone_query)
    response = ollama.chat(
        model=MODEL_NAME,
        messages=template,
        options={"temperature": 0.1}
    )
    answer = response['message']['content'].lower()
    print(f"Answer: {answer}")

    try:
        if 'yes' in answer:
            return True
        elif 'no' in answer:
            return False
    except Exception as e:
        print(f"Error occurred in web search router: {e}")
        return Web_search_router(standalone_query)

def Run_Web_RAG(chat_history, original_user_query, web_rewrite_query):
    # print(web_rewrite_query)
    search_result = client.search(
                    web_rewrite_query, 
                    include_answer = True, 
                    )
    print(search_result)

    search_result.pop('query', None)
    search_result.pop('follow_up_questions', None)
    search_result.pop('images', None)
    search_result.pop('response_time', None)
    search_result['summary'] = search_result.pop('answer')
    
    url_list = []
    title_list = []
    
    for result in search_result.get('results', []):
        result.pop('score', None)
        result.pop('raw_content', None)
        url = result.pop('url', None)
        title = result.get('title')
        
        url_list.append(url)
        title_list.append(title)

    markdown_links = "\n\n"
    for title, url in zip(title_list, url_list):
        markdown_links += f"[{title}]({url})\n\n"
        
    template = chat_history.copy()[:-1] # remove the latest user message
    template = template + response_templates.WEB_RESPONSE_TEMPLATE(search_result, original_user_query)

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

    yield markdown_links
    full_response += markdown_links
    chat_history.append({"role": "assistant", "content": full_response})
