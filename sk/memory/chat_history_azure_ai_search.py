import json
from datetime import datetime
from dataclasses import dataclass
from typing import Annotated
from semantic_kernel.data import (
    VectorStoreRecordCollection,
    VectorStoreRecordDataField,
    VectorStoreRecordKeyField,
    vectorstoremodel
)
from semantic_kernel.contents import (
    ChatMessageContent,
    ChatHistoryTruncationReducer
)
from semantic_kernel.connectors.memory.azure_ai_search import AzureAISearchStore


@vectorstoremodel
@dataclass
class ChatHistoryModel:
    session_id: Annotated[str, VectorStoreRecordKeyField]
    user_id: Annotated[str, VectorStoreRecordDataField(is_filterable=True)]
    messages: Annotated[str, VectorStoreRecordDataField(is_filterable=True)]
    timestamp: Annotated[str, VectorStoreRecordDataField(is_filterable=True)]


class ChatHistoryInAzureAISearch(ChatHistoryTruncationReducer):

    session_id: str | None = None
    user_id: str | None = None
    store: AzureAISearchStore
    collection: VectorStoreRecordCollection | None = None

    async def create_collection(self, collection_name: str) -> None:
        collection = self.store.get_collection(
            collection_name=collection_name,
            data_model_type=ChatHistoryModel
        )
        self.collection = collection
        await self.collection.create_collection_if_not_exists()

    async def store_messages(self) -> None:
        if not self.is_session_info_set():
            raise ValueError(
                "Session info is not set.")
        serialized_messages = json.dumps([msg.model_dump() for msg in self.messages])
        if self.collection:
            await self.collection.upsert(
                ChatHistoryModel(
                    session_id=self.session_id,
                    user_id=self.user_id,
                    messages=serialized_messages,
                    timestamp=datetime.now().isoformat()
                )
            )

    async def read_messages(self) -> None:
        if self.collection:
            record = await self.collection.get(self.session_id)
            if record:
                message_list = json.loads(record.messages)
                for message in message_list:
                    # deserialized_message = ChatMessageContent.model_validate(message)
                    deserialized_message = ChatMessageContent.model_validate(message)
                    self.messages.append(deserialized_message)

    def set_session_info(self, session_id: str, user_id: str) -> None:
        self.session_id = session_id
        self.user_id = user_id

    def is_session_info_set(self) -> bool:
        return self.session_id is not None and self.user_id is not None
