import azure.functions as func
from blueprints.http_chat_blueprint import chat_bp
from azurefunctions.extensions.http.fastapi import Request, JSONResponse

app = func.FunctionApp()

app.register_functions(chat_bp)
