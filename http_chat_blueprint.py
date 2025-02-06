import logging
import azure.functions as func
from azure_openai_service import AzureOpenAIService

chat_bp = func.Blueprint()


@chat_bp.route(route="chat", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def chat(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        prompt = req_body.get("prompt")
        openai_service = AzureOpenAIService()
        content: str | None = openai_service.chat_create_completion(prompt)

        return func.HttpResponse(content, status_code=200)

    return func.HttpResponse(
        "Please enter a valid prompt",
        status_code=400,
    )
