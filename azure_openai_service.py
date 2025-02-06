import os
import openai
from openai.types.chat import ChatCompletionMessageParam
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


class AzureOpenAIService:
    def __init__(self: "AzureOpenAIService") -> None:
        if not hasattr(self, "__client"):
            self.__initialize()
            self.__client = openai

    def chat_create_completion(self: "AzureOpenAIService", prompt: str) -> str:
        system_message: str = os.getenv(
            "SYSTEM_MESSAGE", "You are a helpful assistant."
        )
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
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

        if not content:
            return "Nothing has responded."
        else:
            return content

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
