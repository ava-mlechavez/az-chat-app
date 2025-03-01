import azure.functions as func
from blueprints.http_chat_blueprint import chat_bp
from blueprints.http_search_bp import simple_search_bp

app = func.FunctionApp()

app.register_blueprint(chat_bp)
# app.register_blueprint(simple_search_bp)
