import azure.functions as func
from blueprints.http_chat_blueprint import chat_bp
from blueprints.http_search_bp import simple_search_bp

app = func.FunctionApp()

app.register_functions(chat_bp)
app.register_functions(simple_search_bp)
