import os
import logging
import asyncio
import base64
import azure.functions as func
from azurefunctions.extensions.http.fastapi import Request, StreamingResponse
from azure.identity import ManagedIdentityCredential, AzureCliCredential, get_bearer_token_provider

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase

from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistory

from semantic_kernel.contents import ChatMessageContent, TextContent, ImageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from azure.search.documents import SearchClient

from plugins.hotel_vector_search_plugin import HotelVectorSearchPlugin


def get_credential() -> ManagedIdentityCredential | AzureCliCredential:
    client_id = os.getenv("AZURE_CLIENT_ID")

    if client_id:
        return ManagedIdentityCredential(client_id=client_id)

    return AzureCliCredential()


chat_service = AzureChatCompletion(
    service_id="sample_chat",
    deployment_name="gpt-4o",
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    ad_token_provider=get_bearer_token_provider(
        get_credential(),
        "https://cognitiveservices.azure.com/.default"
    ),
    api_version=os.getenv("OPENAI_API_VERSION", "")
)

history = ChatHistory(
    system_message=f"""
        You are a customer support assistant responsible for recommending hotels based on customer queries.
        When the question is not clear. generate a standalone question. When a user asks for a hotel recommendation, you must reply with accuracy using the HotelVectorSearch plugin. Each object contains key details such as the id, hotel name, category, city, state, and description. Your task is to:

        - Use the `description` field from the list of objects to understand the features and amenities of each hotel.
        - Summarize your answer based on the description.
        - Format the hotel suggestions into a clear, concise, and user-friendly response.
        - Present the information in a way that is easy for the customer to understand, emphasizing the details that are most relevant to their query (such as location, category, and description).
        - If the customer's question is unclear, ask follow-up questions to gather more details about their preferences, such as location, budget, or amenities.
        - **Do not answer any questions that are not related to hotels and outside the knowledge base. If it's a greeting greet them. If the question is not about hotels, politely inform the user that you can only assist with hotel-related inquiries.**
    """
)

kernel = Kernel()
kernel.add_service(chat_service)

# kernel.add_plugin(
#     ImageToQuestionPlugin(kernel=kernel, chat_history=history),
#     plugin_name="ImageToQuestion"
# )

search_client = SearchClient(
    endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT", ""),
    index_name=os.getenv("INDEX_NAME", ""),
    credential=get_credential()
)
kernel.add_plugin(
    HotelVectorSearchPlugin(search_client=search_client),
    plugin_name="HotelVectorSearch"
)


logging.basicConfig(
    format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("kernel").setLevel(logging.DEBUG)

history = ChatHistory(
    system_message="You are a customer support assistant responsible for recommending hotels based on customer queries"
)

bp = func.Blueprint()


async def collect_and_stream(response):
    content = ""

    queue = asyncio.Queue()

    async def collect_content():
        nonlocal content
        async for chunk in response:
            if chunk.content:
                content += chunk.content  # Collect content for later use
                # Put chunk into the queue for streaming
                await queue.put(chunk.content)

        # Indicate the end of the stream
        await queue.put(None)  # None will signal the end of the stream

    async def stream_response():
        while True:
            chunk = await queue.get()
            if chunk is None:  # End of stream
                break
            yield chunk  # Yield chunk to the client

    # Start collecting content asynchronously
    await asyncio.create_task(collect_content())

    # Return the stream to send to the client
    # Return both the stream and the collected content
    return stream_response(), content


@bp.route(route="semantic-kernel-chat", methods=[func.HttpMethod.POST], auth_level=func.AuthLevel.ANONYMOUS)
async def semantic_kernel_chat(req: Request):
    form_data = await req.form()
    prompt = form_data.get("prompt")
    file = form_data.get("file")

    if not file:
        history.add_message(
            ChatMessageContent(
                role=AuthorRole.USER,
                content=prompt
            )
        )
    else:
        file_content = await file.read()
        encoded_image = base64.b64encode(file_content).decode("ascii")
        file_name = file.filename
        file_extension = os.path.splitext(file_name)[1].replace('.', '')
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

    chat_completion: AzureChatCompletion = kernel.get_service(
        type=ChatCompletionClientBase
    )
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    response = chat_completion.get_streaming_chat_message_content(
        kernel=kernel,
        chat_history=history,
        settings=execution_settings,
    )

    response_stream, content = await collect_and_stream(response)
    history.add_message(
        ChatMessageContent(
            role=AuthorRole.ASSISTANT,
            content=content
        )
    )

    return StreamingResponse(response_stream, media_type="text/event-stream")
