import azure.functions as func
from azurefunctions.extensions.http.fastapi import Request, StreamingResponse
import openai
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

sample_bp = func.Blueprint()

@sample_bp.route(
    route="sample", methods=[func.HttpMethod.GET], auth_level=func.AuthLevel.ANONYMOUS
)
async def sample(req: func.http.HttpRequest):
    client = openai.AsyncAzureOpenAI(
        azure_endpoint=openai.azure_endpoint,
        azure_ad_token_provider= get_bearer_token_provider(DefaultAzureCredential()),
        api_version=os.environ["OPENAI_API_VERSION"]
    )

    completion = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Say this is a test."},
        ],
    )
