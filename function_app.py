import azure.functions as func
from http_chat_blueprint import chat_bp

app = func.FunctionApp()

app.register_functions(chat_bp)

