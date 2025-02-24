import os
import openai
import logging
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

class AzureOpenAIService:
    @property
    def client(self):
        return self.__client

    def __init__(self: "AzureOpenAIService") -> None:
        if not hasattr(self, "__client"):
            self.__initialize()
            self.__client = openai

    def chat(
        self: "AzureOpenAIService",
        messages: list[ChatCompletionMessageParam]
    ) -> str:
        try:
            completion = self.__client.chat.completions.create(
                model=os.getenv("DEPLOYMENT_NAME", "gpt-4o"),
                messages=messages,
                max_tokens=800,
                temperature=0.7,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False,
            )
            content = completion.choices[0].message.content

            return content
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
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            openai.azure_ad_token_provider = token_provider
            openai.azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
            openai.api_type = "azure"
            openai.api_version = os.environ["OPENAI_API_VERSION"]
        except Exception as e:
            print("Error during token retrieval", e)
