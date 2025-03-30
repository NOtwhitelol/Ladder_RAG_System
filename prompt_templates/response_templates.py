def DB_RESPONSE_TEMPLATE(source_knowledge, query):
    return [
    {"role": "system", "content": 
f"""You are a helpful AI chatbot on Ladder, an easy-to-use and professional tool for public to build artificial neural network models for data mining.
The following retrieved contents contains parts of informations about Ladder, response to the user's query based on the informations.
---
contents:
{source_knowledge}"""},

    {"role": "user", "content": 
f"""{query}"""}
]





def WEB_RESPONSE_TEMPLATE(search_result, query):
    return [
    {"role": "user", "content": 
f"""You are a helpful AI chatbot on Ladder, an easy-to-use and professional tool for public to build artificial neural network models for data mining.
The following retrieved contents contains informations from the internet, response to the user's query based on the informations.
---
contents:
{search_result}"""},

    {"role": "user", "content":
f"""{query}"""}
]
