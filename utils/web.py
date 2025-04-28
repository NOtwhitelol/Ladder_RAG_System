from tavily import TavilyClient
from prompt_templates import response_templates, router_templates, rewrite_templates
import ollama
import json

from dotenv import load_dotenv
import os
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

client = TavilyClient(api_key=TAVILY_API_KEY)

def Web_query_rewrite(query):
    template = rewrite_templates.WEB_QUERY_REWRITE_TEMPLATE(query)
    response = ollama.chat(
        model="llama3.1",
        messages=template,
        options={
            "temperature": 0.1
        }
    )
    try:
        answer = json.loads(response['message']['content'])
        
        query = answer["query"]
        return query
        
    except (json.JSONDecodeError, KeyError):
        print("Invalid web query rewrite format, retrying...")
        return Web_query_rewrite(query)


def Web_search_router(query):
    template = router_templates.WEB_ROUTER_TEMPLATE(query)
    response = ollama.chat(
        model="llama3.1",
        messages=template,
        options={
            "temperature": 0.1
        }
    )
    answer = response['message']['content'].lower()
    
    try:
        if 'yes' in answer:
            return True
        elif 'no' in answer:
            return False
    except Exception as e:
        print(f"Error occurred in web search router: {e}")
        return Web_search_router(query)

def Run_Web_RAG(chat_history, follow_up_question, query):
    search_result = client.search(
                    query, 
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
    template = template + response_templates.WEB_RESPONSE_TEMPLATE(search_result, follow_up_question)

    # print(template)
    
    response = ollama.chat(
        model="llama3.1",
        messages=template,
        stream=True,
        options={
            "temperature": 0.1
        }
    )

    full_response = ""

    for chunk in response:
        message = chunk['message']['content']
        full_response += message
        yield message
    yield markdown_links

    full_response += markdown_links

    chat_history.append({"role": "assistant", "content": full_response})
