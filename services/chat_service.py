import os
from services.azure_ai_search_service import (
    AzureAISearchService,
)
from services.azure_openai_service import (
    AzureOpenAIService,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam
)


class ChatService:
    async def chat(self: "ChatService", prompt: str, chat_history: list) -> str:
        standalone_question_system_message = self.__create_standalone_question(
            prompt=prompt, chat_history=chat_history)

        openai_service = AzureOpenAIService()
        standalone_question = await openai_service.chat(
            model="gpt-4o-mini",
            messages=[
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=standalone_question_system_message
                ),
            ]
        )

        search_service = AzureAISearchService(
            index_name=os.environ["INDEX_NAME"])
        results = search_service.hybrid_search(query=standalone_question)

        hotels: list[dict] = []

        for result in results:
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

        chat_with_context_system_message = self.__create_chat_with_context(
            prompt=standalone_question,
            chat_history=chat_history,
            context=hotels
        )

        return await openai_service.stream_chat(
            model="gpt-4o",
            messages=[
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=chat_with_context_system_message
                )

             ]
        )


    def __create_standalone_question(self: "ChatService", prompt: str, chat_history: list[ChatCompletionMessageParam]) -> str:
        system_message = os.getenv(
            "STANDALONE_QUESTION_SYSTEM_MESSAGE",
            """
                Given the following chat history and the user's next question,
                rephrase the user's question to be a stand alone question.
                If the chat history is irrelevant or empty, restate the original question.
                Don't add more details to the question.
            """
        )
        return f"""
                {system_message}

                chat history:
                {self.__get_normalized_chat_history(chat_history)}

                follow up question: {prompt}
                standalone question:
            """

    def __create_chat_with_context(self: "ChatService", prompt: str, chat_history: list[ChatCompletionMessageParam], context: list) -> str:
        system_message: str = os.getenv(
            "CHAT_WITH_CONTEXT_SYSTEM_MESSAGE",
            f"""
                You are a polite customer support assistant responsible for recommending hotels based on customer queries. When a user asks for a hotel recommendation, you must reply with accuracy using the context below (as an array of objects). Each object contains key details such as the id, hotel name, category, city, state, and description. Your task is to:

                - Use the `description` field from the list of objects to understand the features and amenities of each hotel.
                - Summarize your answer based on the description.
                - Format the hotel suggestions into a clear, concise, and user-friendly response.
                - Present the information in a way that is easy for the customer to understand, emphasizing the details that are most relevant to their query (such as location, category, and description).
                - If the customer's question is unclear, ask follow-up questions to gather more details about their preferences, such as location, budget, or amenities.
                - **Do not answer any questions that are not related to hotels. If it's a greeting greet them. If the question is not about hotels, politely inform the user that you can only assist with hotel-related inquiries.**
            """,
        )
        return f"""
            {system_message}

            context: {context}

            chat hitory:
            {self.__get_normalized_chat_history(chat_history)}

            user: {prompt}
        """

    def __get_normalized_chat_history(self: "ChatService", chat_history: list[ChatCompletionMessageParam]) -> str:
        return "\n".join([f'{message["content"]}' for message in chat_history])
