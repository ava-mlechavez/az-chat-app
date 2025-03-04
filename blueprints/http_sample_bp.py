import os
import json
import azure.functions as func
from azurefunctions.extensions.http.fastapi import Request, Response
import openai
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

sample_bp = func.Blueprint()

@sample_bp.route(
    route="sample", methods=[func.HttpMethod.GET], auth_level=func.AuthLevel.ANONYMOUS
)
async def sample(req: Request):
    client = openai.AsyncAzureOpenAI(
        azure_endpoint=openai.azure_endpoint,
        azure_ad_token_provider= get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"),
        api_version=os.environ["OPENAI_API_VERSION"]
    )

    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say this is a test."},
        ],
    )

    return Response(
        content=json.dumps({ "message": completion.choices[0].message.content}),
        status_code=200,
        media_type="application/json"
    )
