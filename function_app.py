import azure.functions as func
from http_chat_blueprint import chat_bp
from http_search_bp import simple_search_bp

app = func.FunctionApp()

app.register_functions(chat_bp)
app.register_functions(simple_search_bp)
