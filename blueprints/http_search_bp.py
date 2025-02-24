import logging
import json
from azure.core.credentials import AzureKeyCredential
import azure.functions as func
from azure.search.documents import SearchClient
from azure.search.documents._generated.models import VectorQuery, VectorizableTextQuery
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswParameters,
    SearchIndex,
    SearchFieldDataType,
    SimpleField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.identity import DefaultAzureCredential
from services.azure_openai_service import AzureOpenAIService

simple_search_bp = func.Blueprint()


endpoint = "https://mslearn-ai900-eastus2-basic-search.search.windows.net"
credential = DefaultAzureCredential()

search_index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
search_client = SearchClient(
    endpoint=endpoint, index_name="hotel-vector", credential=credential
)


@simple_search_bp.route(route="sample-embedding", auth_level=func.AuthLevel.ANONYMOUS)
def emb(req: func.HttpRequest) -> func.HttpResponse:
    openai_service = AzureOpenAIService()

    embedding = (
        openai_service.client.embeddings.create(
            input="hello, world", model="text-embedding-3-small"
        )
        .data[0]
        .embedding
    )

    response = {"embedding": embedding}
    return func.HttpResponse(str(response), status_code=200)


@simple_search_bp.route(
    route="simple-search", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS
)
def simple_search(req: func.HttpRequest):
    q = [1, 2, 3]
    r = search_client.search(
        search_text=None,
        vector_queries=[VectorQuery(vector=q, k=2, fields="my_vector")],
    )
    return func.HttpResponse(str(r))


@simple_search_bp.route(
    route="create-index", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS
)
def create_index(req: func.HttpRequest):
    if req.params.get("name"):
        logging.info(f"Hi {req.params.get('name')}")
    try:
        sample_index = SearchIndex(
            name="tiny_vectors",
            fields=[
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(
                    name="category", type=SearchFieldDataType.String, filterable=True
                ),
                SearchField(
                    name="my_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=3,
                    vector_search_profile_name="v_profile",
                ),
            ],
            vector_search=VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="algo",
                        parameters=HnswParameters(metric="cosine"),
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="v_profile", algorithm_configuration_name="algo"
                    )
                ],
            ),
        )

        result = search_index_client.create_index(index=sample_index)
    except Exception as ex:
        return func.HttpResponse(str(ex), status_code=500)

    else:
        return func.HttpResponse(f"{result.name} has been successfully created.")


@simple_search_bp.route(
    route="add-document", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS
)
def add_document(req: func.HttpRequest):
    try:
        search_client.upload_documents(
            documents=[
                {"id": 1, "category": "a", "my_vector": [1, 2, 3]},
                {"id": 2, "category": "a", "my_vector": [1, 1, 3]},
                {"id": 3, "category": "b", "my_vector": [4, 5, 6]},
            ]
        )
    except Exception:
        return func.HttpResponse("Something went wrong", status_code=500)
    else:
        return func.HttpResponse("Success", status_code=201)


@simple_search_bp.route(
    route="hotel-search", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS
)
def hotel_search_post(req: func.HttpRequest):
    try:
        endpoint = "https://mslearn-ai900-eastus2-basic-search.search.windows.net"
        credential = DefaultAzureCredential()
        hotel_search_client = SearchClient(
            # endpoint=endpoint, index_name="hotel-vector", credential=credential
            endpoint=endpoint,
            index_name="hotel-vector",
            credential=AzureKeyCredential(
                ""
            ),
        )

        req_body = req.get_json()
        question = req_body.get("question")
        results = hotel_search_client.search(
            top=10,
            vector_queries=[
                VectorizableTextQuery(
                    text=question, k_nearest_neighbors=5, fields="text_vector"
                )
            ],
        )
        hotels = []
        for result in results:
            hotels.append(
                {
                    "id": result["chunk_id"],
                    "hotelName": result["HotelName"],
                    "category": result["Category"],
                    "city": result["City"],
                    "state": result["State"],
                    "description": result["chunk"],
                }
            )

        return func.HttpResponse(json.dumps({"hotels": hotels}))

    except ValueError:
        return func.HttpResponse("Please enter a prompt")


@simple_search_bp.route(
    route="hotel-search", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS
)
def hotel_search(req: func.HttpRequest):
    try:
        endpoint = "https://mslearn-ai900-eastus2-basic-search.search.windows.net"
        credential = DefaultAzureCredential()
        hotel_search_client = SearchClient(
            endpoint=endpoint,
            index_name="hotel-vector",
            credential=credential,
        )

        question = req.params.get("question")
        if not question:
            question = "luxury"

        results = hotel_search_client.search(
            top=10,
            vector_queries=[
                VectorizableTextQuery(
                    text=question, k_nearest_neighbors=5, fields="text_vector"
                )
            ],
        )
        hotels = []
        for result in results:
            hotels.append(
                {
                    "id": result["chunk_id"],
                    "hotelName": result["HotelName"],
                    "category": result["Category"],
                    "city": result["City"],
                    "state": result["State"],
                    "description": result["chunk"],
                }
            )
        return func.HttpResponse(json.dumps({"hotels": hotels}))

    except ValueError:
        return func.HttpResponse("Please enter a prompt")
