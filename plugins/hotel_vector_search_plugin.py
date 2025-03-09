import os
from typing import Annotated

from semantic_kernel.functions import kernel_function

from azure.identity import ManagedIdentityCredential, AzureCliCredential
from azure.search.documents import SearchClient
from azure.search.documents._generated.models import (
    QueryType,
    VectorQuery,
    VectorizableTextQuery,
)
from azure.search.documents._paging import SearchItemPaged


class HotelVectorSearchPlugin:
    def __init__(self, search_client: SearchClient) -> None:
        self.__client = search_client

    @kernel_function(
        name="search",
        description="Search for documents similar to the given query."
    )
    async def search(
        self,
        query: Annotated[str, "Query to be used for searching"],
    ) -> list[dict]:
        """Search for documents similar to the given query."""

        vector_queries: list[VectorQuery] | None = [
            VectorizableTextQuery(
                text=query,
                k_nearest_neighbors=10,
                fields="text_vector"
            )
        ]

        query_args = {
            "search_text": query,
            "vector_queries": vector_queries,
            "query_type": QueryType.SEMANTIC,
            "semantic_configuration_name": os.environ["SEMANTIC_CONFIGURATION_NAME"],
        }

        # if use_semantic_query:
        #     query_args.update(
        #         {
        #             "query_type": QueryType.SEMANTIC,
        #             "semantic_configuration_name": os.environ["SEMANTIC_CONFIGURATION_NAME"],
        #         }
        #     )

        results = self.__client.search(**query_args)
        hotels: list[dict] = []

        for result in results:
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

    def __get_credential(self) -> ManagedIdentityCredential | AzureCliCredential:
        client_id = os.getenv("AZURE_CLIENT_ID")

        if client_id:
            return ManagedIdentityCredential(client_id=client_id)

        return AzureCliCredential()
