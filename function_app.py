import azure.functions as func
from blueprints.http_semantic_kernel_bp import bp as semantic_kernel_bp
from blueprints.http_sk import bp as sk_bp

app = func.FunctionApp()

app.register_functions(semantic_kernel_bp)
app.register_functions(sk_bp)
