import json
import azure.functions as func
from services.chat_service import ChatService

chat_bp = func.Blueprint()


@chat_bp.route(
    route="chat", methods=[func.HttpMethod.POST], auth_level=func.AuthLevel.ANONYMOUS
)
def chat(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()
        prompt = req_body.get("prompt")
        chat_history = req_body.get("chat_history")

        chat_service = ChatService()
        content = chat_service.chat(prompt=prompt, chat_history=list(chat_history))

        return func.HttpResponse(json.dumps({ "message": content}), status_code=200)
    except ValueError as e:
        return func.HttpResponse(
            str(e),
            status_code=500
        )
