import os
import logging
import base64
from pydantic import BaseModel
import azure.functions as func
from azurefunctions.extensions.http.fastapi import Request, StreamingResponse, JSONResponse
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)
from azure.identity import ManagedIdentityCredential, AzureCliCredential, get_bearer_token_provider
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.contents import ChatMessageContent, TextContent, ImageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from azure.search.documents.indexes.aio import SearchIndexClient
from pydantic import PrivateAttr
from semantic_kernel.functions import kernel_function
from typing import Annotated
from azure.search.documents._generated.models import (
    QueryType,
    VectorQuery,
    VectorizableTextQuery,
)
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
import asyncio

bp = func.Blueprint()


async def collect_and_stream(response):
    content = ""

    queue = asyncio.Queue()

    async def collect_content():
        nonlocal content
        async for chunk in response:
            if chunk.content:
                content += chunk.content
                await queue.put(chunk.content)

        await queue.put(None)

    async def stream_response():
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    await collect_content()

    return stream_response(), content


def get_credential() -> ManagedIdentityCredential | AzureCliCredential:
    client_id = os.getenv("AZURE_CLIENT_ID")

    if client_id:
        return ManagedIdentityCredential(client_id=client_id)

    return AzureCliCredential()


search_index_client = SearchIndexClient(
    endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT", ""),
    credential=get_credential(),  # pyright: ignore
)

gpt_4o_service = AzureChatCompletion(
    service_id="gpt4o",
    deployment_name="gpt-4o",
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    ad_token_provider=get_bearer_token_provider(
        get_credential(), "https://cognitiveservices.azure.com/.default"
    ),
    api_version=os.getenv("OPENAI_API_VERSION", ""),
)

gpt_4o_mini_service = AzureChatCompletion(
    service_id="gpt4omini",
    deployment_name="gpt-4o-mini",
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    ad_token_provider=get_bearer_token_provider(
        get_credential(), "https://cognitiveservices.azure.com/.default"
    ),
    api_version=os.getenv("OPENAI_API_VERSION", ""),
)

ada_embedding_service = AzureTextEmbedding(
    service_id="ada_embedding",
    deployment_name="text-embedding-ada-002",
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    ad_token_provider=get_bearer_token_provider(
        get_credential(), "https://cognitiveservices.azure.com/.default"
    ),
    api_version=os.getenv("OPENAI_API_VERSION", ""),
)

chat_history = ChatHistory(
    system_message=""""
        You are a customer support assistant responsible for recommending hotels based on customer queries.
        When the question is not clear. generate a standalone question. When a user asks for a hotel recommendation, you must reply with accuracy using the HotelVectorSearch plugin. Each object contains key details such as the hotel name, category, city, state, and description. Your task is to:

        - Use the `description` field from the list of objects to understand the features and amenities of each hotel.
        - Summarize your answer based on the description.
        - Format the hotel suggestions into a clear, concise, and user-friendly response.
        - Present the information in a way that is easy for the customer to understand, emphasizing the details that are most relevant to their query (such as location, category, and description).
        - If the customer's question is unclear, ask follow-up questions to gather more details about their preferences, such as location, budget, or amenities.
        - **Do not answer any questions that are not related to hotels and outside the knowledge base. If it's a greeting greet them. If the question is not about hotels, politely inform the user that you can only assist with hotel-related inquiries.**
    """
)


class HotelSearchPlugin:
    def __init__(self, search_index_client: SearchIndexClient) -> None:
        self._search_index_client = search_index_client

    @kernel_function(
        name="search", description="Search for documents similar to the given query."
    )
    async def search(
        self,
        query: Annotated[str, "Query to be used for searching"],
    ) -> list[dict]:
        """Search for documents similar to the given query."""
        try:
            vector_queries: list[VectorQuery] | None = [
                VectorizableTextQuery(
                    text=query, k_nearest_neighbors=10, fields="text_vector"
                )
            ]

            query_args = {
                "search_text": query,
                "vector_queries": vector_queries,
                "query_type": QueryType.SEMANTIC,
                "semantic_configuration_name": os.environ[
                    "SEMANTIC_CONFIGURATION_NAME"
                ],
            }

            search_client = self._search_index_client.get_search_client(
                index_name=os.environ["AZURE_AI_SEARCH_INDEX_NAME"]
            )

            async with search_client:  # pyright: ignore
                # pyright: ignore
                results = await search_client.search(**query_args)
                hotels: list[dict] = []

                async for result in results:
                    hotels.append(
                        {
                            "id": result["Id"],
                            "hotelName": result["HotelName"],
                            "category": result["Category"],
                            "city": result["City"],
                            "state": result["State"],
                            "description": result["chunk"],
                        }
                    )

                return hotels

        except Exception as e:
            logging.error(f"Error in search: {e}")
            return []


class EmailSenderPlugin:
    @kernel_function(
        name="send_email", description="Send an email to the given email address."
    )
    async def send_email(self, email: str, message: str):
        return f"Email sent to {email} with message: {message}"

kernel = Kernel()
kernel.add_service(gpt_4o_service)
kernel.add_service(gpt_4o_mini_service)
kernel.add_service(ada_embedding_service)

kernel.add_plugin(
    HotelSearchPlugin(search_index_client=search_index_client),
    plugin_name="HotelSearchPlugin")
kernel.add_plugin(
    EmailSenderPlugin(),
    plugin_name="EmailSender"
)


@bp.route(
    route="sk-demo",
    methods=[func.HttpMethod.POST],
    auth_level=func.AuthLevel.ANONYMOUS
)
async def sk_demo(req: Request) -> JSONResponse:
    try:
        form_data = await req.form()
        prompt = form_data.get("prompt")
        file = form_data.get("file")

        response: ChatMessageContent
        execution_settings: AzureChatPromptExecutionSettings
        if file is None:
            if prompt is None:
                return JSONResponse({"message": "Prompt is required."})

            chat_completion: AzureChatCompletion = kernel.get_service(
                service_id="gpt4omini")
            execution_settings = kernel.get_prompt_execution_settings_from_service_id(
                service_id="gpt4omini")

            chat_history.add_message(
                message=ChatMessageContent(
                    role=AuthorRole.USER, content=prompt)
            )

        else:
            chat_completion: AzureChatCompletion = kernel.get_service(
                service_id="gpt4o")
            execution_settings = kernel.get_prompt_execution_settings_from_service_id(
                service_id="gpt4o")

            file_content = await file.read()
            encoded_image = base64.b64encode(file_content).decode("ascii")
            filename = file.filename
            file_extension: str = ""
            if filename:
                file_extension = os.path.splitext(filename)[1].replace(".", "")

            chat_history.add_message(
                message=ChatMessageContent(
                    role=AuthorRole.USER,
                    items=[
                        TextContent(
                            text="""
                                Analyze the features and amenities in this image, and generate a conceptual similarity that can be used for a vector search. Based on this analysis, create a standalone question relevant to the image. Do not include the question in the response. Instead, invoke the hotel vector search plugin using the generated question.
                            """
                        ),
                        ImageContent(
                            data_uri=f"data:image/{file_extension};base64,{encoded_image}"
                        ),
                    ],
                )
            )

        execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
        response = chat_completion.get_streaming_chat_message_content(
            kernel=kernel,
            chat_history=chat_history,
            settings=execution_settings,
        )
        response_stream, content = await collect_and_stream(response)
        chat_history.add_message(
            message=ChatMessageContent(
                role=AuthorRole.ASSISTANT, content=content)
        )
        return StreamingResponse(response_stream, media_type="text/event-stream")
    except Exception as e:
        return JSONResponse({"message": str(e)})
