from typing import Annotated
from semantic_kernel.kernel import Kernel
from semantic_kernel.contents import ChatMessageContent, AuthorRole, TextContent, ImageContent
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory


class ImageToQuestionPlugin:
    def __init__(self, kernel: Kernel, chat_history: ChatHistory):
        self.__kernel = kernel
        self.__chat_history = chat_history

    @kernel_function(
        name="generate_question_from_image",
        description="Generate a standalone question based on the image data uri base64 encoded image"
    )
    async def generate_question_from_image(
        self,
        data_uri: Annotated[
            str,
            # "A data URI, a base64 encoded image (e.g. data:image/{file_extension};base64,{encoded_image})"
            "A data URI, a base64 encoded image"
        ]
    ) -> str:
        """Generate a standalone question based on the base64 encoded image"""

        try:
            chat_service: AzureChatCompletion = self.__kernel.get_service(
                type=ChatCompletionClientBase)

            prompt = f"Describe this picture and understand its features. Use the description to generate a relevant standalone question, but do not include the question in the response."

            self.__chat_history.add_message(
                message=ChatMessageContent(
                    role=AuthorRole.USER,
                    items=[
                        TextContent(text=prompt),
                        ImageContent(data_uri=data_uri)
                    ]
                )
            )

            prompt_execution_settings = AzureChatPromptExecutionSettings()

            response = await chat_service.get_chat_message_content(
                chat_history=self.__chat_history,
                settings=prompt_execution_settings
            )

            return response.content

        except Exception as e:
            return f"Error occurred: {str(e)}"
