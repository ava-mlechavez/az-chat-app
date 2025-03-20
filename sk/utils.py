import os
import logging
from azure.identity import ManagedIdentityCredential, AzureCliCredential

from azure.identity import get_bearer_token_provider
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)
from semantic_kernel import Kernel

from azure.search.documents.indexes.aio import SearchIndexClient
from semantic_kernel.connectors.memory.azure_ai_search import AzureAISearchStore

from sk.memory.chat_history_azure_ai_search import ChatHistoryInAzureAISearch
from sk.plugins.hotel_vector_search_plugin import HotelVectorSearchPlugin


def get_credential() -> ManagedIdentityCredential | AzureCliCredential:
    client_id = os.getenv("AZURE_CLIENT_ID")

    if client_id:
        return ManagedIdentityCredential(client_id=client_id)

    return AzureCliCredential()


def initialize_search_index_client() -> SearchIndexClient:
    return SearchIndexClient(
        endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT", ""),
        credential=get_credential(),  # pyright: ignore
    )


def initialize_semantic_kernel(search_index_client: SearchIndexClient) -> Kernel:
    gpt4omini_service = AzureChatCompletion(
        service_id="gp4omini_chat",
        deployment_name="gpt-4o-mini",
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        ad_token_provider=get_bearer_token_provider(
            get_credential(), "https://cognitiveservices.azure.com/.default"
        ),
        api_version=os.getenv("OPENAI_API_VERSION", ""),
    )

    gpt4o_service = AzureChatCompletion(
        service_id="gp4o_chat",
        deployment_name="gpt-4o",
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

    kernel = Kernel()
    kernel.add_service(gpt4omini_service)
    kernel.add_service(gpt4o_service)
    kernel.add_service(ada_embedding_service)

    kernel.add_plugin(
        HotelVectorSearchPlugin(search_index_client=search_index_client),
        plugin_name="HotelVectorSearch",
    )

    logging.basicConfig(
        format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("kernel").setLevel(logging.DEBUG)

    return kernel


def initialize_store(search_index_client: SearchIndexClient) -> AzureAISearchStore:
    return AzureAISearchStore(search_index_client=search_index_client)


async def initialize_chat_history(
    store: AzureAISearchStore,
) -> ChatHistoryInAzureAISearch:

    history = ChatHistoryInAzureAISearch(
        store=store, target_count=30, threshold_count=30
    )

    await history.create_collection(collection_name="chat-history")
    await history.read_messages()

    if len(history) == 0:
        history.add_system_message(
            """
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

    return history
