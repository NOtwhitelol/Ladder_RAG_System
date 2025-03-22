from flask import Flask, render_template, request, Response
from utils import base, database, web

def RAG_Run(chat_history, follow_up_question, standalone_query):
    print("running")
    if base.Router(standalone_query):
        print("Closed domain query\n")
        if database.DB_search_router(standalone_query):
            print("Using database...\n")
            return database.Run_DB_RAG(chat_history, follow_up_question, standalone_query)
        else:
            chat_history.append({"role": "assistant", "content": "Unable to answer."})
            return "Unable to answer."
    else:
        print("Open domain query\n")
        if web.Web_search_router(standalone_query):
            query = web.Web_query_rewrite(standalone_query)
            print("Rewritten web query:\n" + query)
            print("Searching web...\n")
            return web.Run_Web_RAG(chat_history, follow_up_question, query)
        else:
            print("Direct answer...\n")
            return base.Run_Direct_RAG(chat_history, follow_up_question, standalone_query)




chat_history = [{"role": "system", 
                    "content": """You are a helpful AI assistant on Ladder, an easy-to-use and professional tool for public to build visualize artificial neural network models.
                    Provide clear and concise explanations, assist with troubleshooting, and guide users through Ladder's features, including model creation, training, and evaluation.
                    Always prioritize accuracy and relevance, ensuring responses are practical and actionable. Avoid speculation and only provide information based on Ladder's capabilities and machine learning principles."""
                }]




app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history

    user_message = request.form['message']

    if user_message.strip() == "/history":
        print(chat_history)
        return chat_history
    
    chat_history.append({"role": "user", "content": user_message})

    standalone_query = base.standalone_query_rewrite(chat_history, user_message)

    return Response(RAG_Run(chat_history, user_message, standalone_query), content_type='text/plain')

if __name__ == '__main__':
    app.run(debug=True)