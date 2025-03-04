import os
from azure.identity import ManagedIdentityCredential, AzureCliCredential
from azure.search.documents import SearchClient
from azure.search.documents._generated.models import (
    QueryType,
    VectorQuery,
    VectorizedQuery,
    VectorizableTextQuery,
)
from azure.search.documents._paging import SearchItemPaged


class AzureAISearchService:
    def __init__(self: "AzureAISearchService", index_name: str):
        if not hasattr(self, "__client"):
            self.__client = SearchClient(
                endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT", ""),
                index_name=index_name,
                credential=self.__get_credential()
            )

    def keyword_search(self: "AzureAISearchService", query: str) -> SearchItemPaged[dict]:
        return self.__client.search(search_text=query)

    def vector_search(
        self: "AzureAISearchService",
        embedding: list[float],
    ) -> SearchItemPaged[dict]:
        return self.__client.search(
            search_text=None,
            vector_queries=[
                VectorizedQuery(
                    vector=embedding, k_nearest_neighbors=3, fields="text_vector"
                )
            ],
        )

    def hybrid_search(
        self: "AzureAISearchService",
        query: str,
        use_semantic_query: bool = True,
        **kwargs: dict
    ) -> SearchItemPaged[dict]:
        # k_nearest_neighbors = 10 if use_semantic_query else 3
        vector_queries: list[VectorQuery] | None = [
            VectorizableTextQuery(
                text=query,
                # k_nearest_neighbors= k_nearest_neighbors,
                fields="text_vector"
            )
        ]

        query_args = {
            "search_text": query,
            # "top": 3,
            "vector_queries": vector_queries
        }

        if use_semantic_query:
            query_args.update(
                {
                    "query_type": QueryType.SEMANTIC,
                    "semantic_configuration_name": os.environ["SEMANTIC_CONFIGURATION_NAME"],
                }
            )

        return self.__client.search(**query_args, **kwargs)

    def __get_credential(self: "AzureAISearchService") -> ManagedIdentityCredential | AzureCliCredential:
        client_id = os.getenv("AZURE_CLIENT_ID")

        if client_id:
            return ManagedIdentityCredential(client_id=client_id)

        return AzureCliCredential()
