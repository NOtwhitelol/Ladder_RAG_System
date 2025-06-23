import os
import json

import ollama

from dotenv import load_dotenv
from tavily import TavilyClient
from prompt_templates import response_templates, router_templates, rewrite_templates


# --- Configuration ---
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MODEL_NAME = "phi4:14b"

client = TavilyClient(api_key=TAVILY_API_KEY)


# --- Utility Functions ---
def clean_json_string(s):
    """Remove code block formatting from LLM-generated JSON"""
    s = s.strip()
    if s.startswith("```json"):
        s = s.removeprefix("```json").strip()
    if s.endswith("```"):
        s = s.removesuffix("```").strip()
    return s


# --- Web Query Pipeline ---
def rewrite_web_query(standalone_query) -> str:
    """Rewrite a query for web search."""
    template = rewrite_templates.WEB_QUERY_REWRITE_TEMPLATE(standalone_query)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=template,
        options={"temperature": 0.1}
    )
    rewritten_web_query = clean_json_string(response['message']['content'])
    print(f"Web Rewrite Query: {rewritten_web_query}")

    try:
        answer = json.loads(rewritten_web_query)
        return answer["query"]
    except (json.JSONDecodeError, KeyError):
        return rewritten_web_query  # Return raw string if JSON parse fails

def should_use_web_search(standalone_query) -> bool:
    """Determine if the query should be answered using web search."""
    template = router_templates.WEB_ROUTER_TEMPLATE(standalone_query)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=template,
        options={"temperature": 0.1}
    )
    answer = response['message']['content'].lower()

    if 'yes' in answer:
        return True
    else:
        return False

def run_web_rag(chat_history, original_user_query, web_rewrite_query):
    """Execute RAG pipeline using web search results."""
    search_result = client.search(web_rewrite_query, include_answer = True)
    print(search_result)

    # Clean unnecessary metadata
    search_result.pop('query', None)
    search_result.pop('follow_up_questions', None)
    search_result.pop('images', None)
    search_result.pop('response_time', None)
    search_result['summary'] = search_result.pop('answer')
    
    # Extract useful URLs and titles for reference
    url_list = []
    title_list = []
    for result in search_result.get('results', []):
        result.pop('score', None)
        result.pop('raw_content', None)
        url = result.pop('url', None)
        title = result.get('title')
        url_list.append(url)
        title_list.append(title)

    # Format references as Markdown links
    markdown_links = "\n\n"
    for title, url in zip(title_list, url_list):
        markdown_links += f"[{title}]({url})\n\n"
        
    template = chat_history.copy()[:-1] # remove the latest user message
    template = template + response_templates.WEB_RESPONSE_TEMPLATE(search_result, original_user_query)

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

    yield markdown_links
    full_response += markdown_links
    chat_history.append({"role": "assistant", "content": full_response})
