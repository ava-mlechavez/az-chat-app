import json
import azure.functions as func
from azurefunctions.extensions.http.fastapi import Request, StreamingResponse, Response, JSONResponse
from services.chat_service import ChatService
import base64

chat_bp = func.Blueprint()


async def stream_processor(response):
    async for chunk in response:
        if len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta is not None and delta.content:
                yield delta.content


@chat_bp.route(
    route="chat", methods=[func.HttpMethod.POST], auth_level=func.AuthLevel.ANONYMOUS
)
async def chat(req: Request):
    try:
        req_body = await req.json()
        prompt = req_body.get("prompt")
        chat_history = req_body.get("chat_history")

        chat_service = ChatService()
        response = await chat_service.chat(prompt=prompt, chat_history=list(chat_history))

        return StreamingResponse(stream_processor(response), media_type="text/event-stream")
    except ValueError as e:
        return Response(
            str(e),
            status_code=500
        )


@chat_bp.route(
    route="upload-image", methods=[func.HttpMethod.POST], auth_level=func.AuthLevel.ANONYMOUS
)
async def upload_image(req: Request):
    try:
        form_data = await req.form();
        file = form_data.get("file")
        chat_history = form_data.get("chat_history")

        if not file:
            return JSONResponse({ "message": "No file uploaded" })

        file_content = await file.read()
        encoded_image = base64.b64encode(file_content).decode("ascii")

        if chat_history:
            chat_history = json.loads(chat_history)
        else:
            chat_history = []

        chat_service = ChatService()
        response = await chat_service.analyza_image(encoded_image=encoded_image, chat_history=chat_history)

        return StreamingResponse(stream_processor(response), media_type="text/event-stream")
    except ValueError as e:
        return Response(
            str(e),
            status_code=500
        )
