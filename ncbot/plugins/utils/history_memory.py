from ncbot.plugins.utils.history import MemoryHistoryUtil
from langchain.memory import ChatMessageHistory


class InMemoryHistoryUtil(MemoryHistoryUtil):

    def __init__(self) -> None:
        super().__init__()
        self.__memories = {}

    def _save_to_memory(self, conversation_token, history):
        index_key = super()._get_index_key(conversation_token)
        self.__memories[index_key] = history

    def _get_from_memory(self, conversation_token):
        if not super()._isStore():
            return None
        index_key = super()._get_index_key(conversation_token)
        if not index_key in self.__memories:
            return None
        return self.__memories[index_key]
    

    def clear_memory(self, conversation_token):
        del self.__memories[super()._get_index_key(conversation_token)]