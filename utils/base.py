import ollama
from prompt_templates import response_templates, router_templates, rewrite_templates
import json

def standalone_query_rewrite(chat_history, query):
    response = ollama.chat(
        model="llama3.1",
        messages=rewrite_templates.STANDALONE_QUESTION_REWRITE_TEMPLATE(chat_history, query),
        options={
            "temperature": 0.1
        }
    )
    print(f"Rewrite Query: {response['message']['content']}")

    try:
        data = json.loads(response['message']['content'])
        return data["question"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error parsing rewritten query: {e}")
        return query  # Return the original query as a fallback

def Router(query):
    template = router_templates.ROUTER_TEMPLATE(query)
    
    response = ollama.chat(
        model="llama3.1",
        messages=template,
        options={
            "temperature": 0.1
        }
    )
    answer = response['message']['content']
    try:
        if 'vectorstore' in answer:
            return True
        elif 'web search' in answer:
            return False
    
    except Exception as e:
        print(f"Error occurred in router: {e}")
        return Router(query)


def Run_Direct_RAG(chat_history, follow_up_question, standalone_query):
    template = chat_history.copy()[:-1] # remove the latest user message
    template.append({"role": "user", "content": follow_up_question})
    
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
    
    chat_history.append({"role": "assistant", "content": full_response})
    