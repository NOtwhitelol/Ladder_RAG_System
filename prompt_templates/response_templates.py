def DB_RESPONSE_TEMPLATE(source_knowledge, query):
    return [{"role": "user", "content": 
f"""You are a helpful AI chatbot on Ladder, an easy-to-use and professional tool for public to build artificial neural network models for data mining.
The following retrieved contents contains parts of informations about Ladder, response to the user's query based on the information.
Note: Not all the contents were needed to response, contents might contain unnecessary informations.
If you're not certain of the answer, just say you don't know.

---
contents:
{source_knowledge}

question:
{query}"""}]





def WEB_RESPONSE_TEMPLATE(search_result, query):
    return [{"role": "user", "content": 
f"""You are an expert in machine learning. Answer the question using the web searching contents below.
---
contents:
{search_result}

question:
{query}"""},
]
