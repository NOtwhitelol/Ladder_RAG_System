from flask import Flask, render_template, request, Response

from utils import base, database, web

app = Flask(__name__)

chat_history = [{"role": "system", 
"content": """You are a helpful AI assistant on Ladder, an easy-to-use and professional tool for public to build visualize artificial neural network models.
Provide clear and concise explanations, assist with troubleshooting, and guide users through Ladder's features, including model creation, training, and evaluation.
Always prioritize accuracy and relevance, ensuring responses are practical and actionable. Avoid speculation and only provide information based on Ladder's capabilities and machine learning principles."""
}]

def run_rag_pipeline(chat_history, original_user_query):
    standalone_query = base.generate_standalone_query(chat_history, original_user_query)

    if base.is_close_domain_query(standalone_query):
        print("[Closed domain query]\n")
        related_section_indices = database.search_db_sections(standalone_query)

        if len(related_section_indices) > 0:
            print("Using database...\n")
            return database.run_db_rag(chat_history, original_user_query, related_section_indices)
        else:
            chat_history.append({"role": "assistant", "content": "Unable to answer."})
            return "Unable to answer."
    else:
        print("[Open domain query]\n")
        if web.should_use_web_search(standalone_query):
            web_rewrite_query = web.rewrite_web_query(standalone_query)
            print("Searching web...\n")
            return web.run_web_rag(chat_history, original_user_query, web_rewrite_query)
        else:
            print("Direct answer...\n")
            return base.direct_response(chat_history, original_user_query)

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

    return Response(run_rag_pipeline(chat_history, user_message), content_type='text/plain')

if __name__ == '__main__':
    app.run(debug=True)