import os
import logging
from typing import Annotated
from pydantic import PrivateAttr

from semantic_kernel.functions import kernel_function

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents._generated.models import (
    QueryType,
    VectorQuery,
    VectorizableTextQuery,
)

class HotelVectorSearchPlugin:
    _search_index_client: SearchIndexClient = PrivateAttr()

    def __init__(self, search_index_client: SearchIndexClient) -> None:
        self._search_index_client = search_index_client


    @kernel_function(
        name="search",
        description="Search for documents similar to the given query."
    )
    async def search(
        self,
        query: Annotated[str, "Query to be used for searching"],
    ) -> list[dict]:
        """Search for documents similar to the given query."""
        try:
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

            search_client = self._search_index_client.get_search_client(
            index_name=os.environ["AZURE_AI_SEARCH_INDEX_NAME"])

            async with search_client:
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
