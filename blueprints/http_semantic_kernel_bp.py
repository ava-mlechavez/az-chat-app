import os
import logging
import base64
import azure.functions as func
from azurefunctions.extensions.http.fastapi import Request, StreamingResponse, JSONResponse

from semantic_kernel import kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatMessageContent, TextContent, ImageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

from sk.utils import (
    initialize_semantic_kernel,
    initialize_search_index_client,
    initialize_store,
    initialize_chat_history
)
from utils import collect_and_stream


bp = func.Blueprint()


search_index_client = initialize_search_index_client()
kernel = initialize_semantic_kernel(search_index_client=search_index_client)
store = initialize_store(search_index_client=search_index_client)

GPT4OMINI_SERVICE_ID = "gp4omini_chat"
GPT4O_SERVICE_ID = "gp4o_chat"

@bp.route(route="semantic-kernel-chat", methods=[func.HttpMethod.POST], auth_level=func.AuthLevel.ANONYMOUS)
async def semantic_kernel_chat(req: Request):
    try:
        form_data = await req.form()
        prompt = form_data.get("prompt")
        file = form_data.get("file")
        session_id = req.headers["X-Chat-Session-Id"]

        history = await initialize_chat_history(store=store)
        history.set_session_info(
            session_id=session_id,
            user_id="user"
        )

        # response_stream = None
        # async with store:
        chat_completion: AzureChatCompletion
        execution_settings: AzureChatPromptExecutionSettings
        if not file:
            chat_completion = kernel.get_service(GPT4OMINI_SERVICE_ID)
            execution_settings = kernel.get_prompt_execution_settings_from_service_id(
                GPT4OMINI_SERVICE_ID)
            history.add_message(
                message=ChatMessageContent(
                    role=AuthorRole.USER,
                    content=prompt
                )
            )

        else:
            chat_completion = kernel.get_service(GPT4O_SERVICE_ID)
            execution_settings = kernel.get_prompt_execution_settings_from_service_id(
                GPT4O_SERVICE_ID)
            file_content = await file.read()
            encoded_image = base64.b64encode(file_content).decode("ascii")
            file_name = file.filename
            file_extension = os.path.splitext(
                file_name)[1].replace('.', '')
            history.add_message(
                message=ChatMessageContent(
                    role=AuthorRole.USER,
                    items=[
                        TextContent(
                            text=f"""
                                    Analyze the features and amenities in this image, and generate a conceptual similarity that can be used for a vector search. Based on this analysis, create a standalone question relevant to the image. Do not include the question in the response. Instead, invoke the hotel vector search plugin using the generated question.
                                """
                        ),
                        ImageContent(
                            data_uri=f'data:image/{file_extension};base64,{encoded_image}')
                    ],
                )
            )

        execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

        response = chat_completion.get_streaming_chat_message_content(
            kernel=kernel,
            chat_history=history,
            settings=execution_settings,
        )

        response_stream, content = await collect_and_stream(response)

        history.add_message(
            message=ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                content=content
            )
        )
        await history.store_messages()
        await history.reduce()

        return StreamingResponse(response_stream, media_type="text/event-stream")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
