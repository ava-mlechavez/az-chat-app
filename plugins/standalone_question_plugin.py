from typing import Annotated
from semantic_kernel.kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.contents import ChatHistory
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.contents import ChatMessageContent


class StandaloneQuestionPlugin:
    def __init__(self, kernel: Kernel, chat_history: ChatHistory):
        self.__chat_service: AzureChatCompletion = kernel.get_service(
            type=ChatCompletionClientBase)
        self.__chat_history = chat_history

    @kernel_function(
        name="create_standalone_question",
        description="Create a standalone question."
    )
    async def create_standalone_question(
        self,
        prompt: Annotated[str, "Used to create a standalone question"],

    ) -> str:
        """Create a standalone question."""

        self.__chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.USER,
                content=prompt
            )
        )

        response = await self.__chat_service.get_chat_message_content(
            chat_history=self.__chat_history,
            settings=AzureChatPromptExecutionSettings()
        )
        return response.content
