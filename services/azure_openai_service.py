import os
import openai
import logging
from openai.types.chat import (
    ChatCompletionMessageParam
)

from azure.identity import ManagedIdentityCredential, AzureCliCredential, get_bearer_token_provider

class AzureOpenAIService:
    @property
    def client(self):
        return self.__client

    def __init__(self: "AzureOpenAIService") -> None:
        if not hasattr(self, "__client"):
            self.__initialize()
            self.__client = openai.AsyncAzureOpenAI(
                azure_endpoint=openai.azure_endpoint,
                azure_ad_token_provider=openai.azure_ad_token_provider,
                api_version=openai.api_version
            )

    async def chat(
        self: "AzureOpenAIService",
        model: str,
        messages: list[ChatCompletionMessageParam]
    ):
        try:
            completion = await self.__client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=800,
                temperature=0.7,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error("Error during chat", e)
            return ""

    async def stream_chat(
            self: "AzureOpenAIService",
            model: str,
            messages: list[ChatCompletionMessageParam]
        ):
            try:
                return await self.__client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    stream=True,
                )
            except Exception as e:
                logging.error("Error during chat", e)
                return ""

    def create_embedding(self: "AzureOpenAIService", input: str) -> list[float]:
        return (
            self.__client.embeddings.create(
                input=input, model="text-embedding-3-small")
            .data[0]
            .embedding
        )

    def __initialize(self: "AzureOpenAIService") -> None:
        try:
            credential = self.__get_credential()
            token_provider = get_bearer_token_provider(
                credential,
                "https://cognitiveservices.azure.com/.default"
            )
            openai.azure_ad_token_provider = token_provider
            openai.azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
            openai.api_type = "azure"
            openai.api_version = os.environ["OPENAI_API_VERSION"]
        except Exception as e:
            print("Error during token retrieval", e)


    def __get_credential(self: "AzureOpenAIService") -> ManagedIdentityCredential | AzureCliCredential:
        client_id = os.getenv("AZURE_CLIENT_ID")

        if client_id:
            return ManagedIdentityCredential(client_id=client_id)

        return AzureCliCredential()
