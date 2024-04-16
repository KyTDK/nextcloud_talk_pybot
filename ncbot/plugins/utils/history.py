
from abc import abstractmethod
from langchain.schema import messages_from_dict, messages_to_dict
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
import ncbot.config as ncconfig


class MemoryHistoryUtil():

    def __init__(self):
        self.max_chat_history = ncconfig.cf.max_chat_history
        self.save_type = ncconfig.cf.save_type

    def _isStore(self):
        return self.max_chat_history != 0

    @abstractmethod
    def _save_to_memory(self, conversation_token, history):
        pass

    @abstractmethod
    def _get_from_memory(self, conversation_token):
        pass

    @abstractmethod
    def clear_memory(self, conversation_token):
        pass

    # def get_memory(self, conversation_token):
    #     dict = self._get_from_memory(conversation_token)
    #     if dict is None or not dict:  # Check for None and empty dictionary
    #         return ConversationBufferMemory(return_messages=True, chat_memory=ChatMessageHistory(messages=[]))
    #     memory_dict = self.__dict_to_message(dict)
    #     history = ChatMessageHistory()
    #     history.messages = history.messages + memory_dict
    #     return ConversationBufferMemory(return_messages=True, chat_memory=history)

    def get_memory(self, conversation_token):
        return self._get_from_memory(conversation_token)
        
    # def save_memory(self, conversation_token, history: ConversationBufferMemory):
    #     chat_memory = history.chat_memory
    #     memory = self.__tuncate_memory(chat_memory)
    #     self._save_to_memory(conversation_token, memory)

    def save_memory(self, conversation_token, history: str):
        self._save_to_memory(conversation_token, self.__tuncate_memory(history))

    def count_tokens_in_dict(self, memory_dict, llm):
        count = 0
        for entry in memory_dict:
            if entry.get('data'):  # Check if 'data' key exists
                data = entry['data']
                if data.get('content'):  # Check if 'content' key exists within 'data'
                    content = data['content']
                    count += llm.get_num_tokens(content)
        return count

    def __tuncate_memory(self, history):
        print("Before: "+str(history))
        llm_gpt3 = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo-0125")
        token_amount = llm_gpt3.get_num_tokens(history)
        TOKEN_LIMIT = ncconfig.cf.max_chat_history
        while token_amount > TOKEN_LIMIT:
            history = history[(token_amount-TOKEN_LIMIT):]
            token_amount = llm_gpt3.get_num_tokens(history)
        print("After: "+str(history))
        return history

    def _get_index_key(self, conversation_token):
        return f'memory_{conversation_token}'

    def __message_to_dict(self, history: ChatMessageHistory):
        return messages_to_dict(history.messages)

    def __dict_to_message(self, load_dict):
        return messages_from_dict(load_dict)
    
from ncbot.plugins.utils.history_memory import InMemoryHistoryUtil
from ncbot.plugins.utils.history_redis import RedisMemoryHistoryUtil

in_memory_util = InMemoryHistoryUtil()
redis_memory_util = RedisMemoryHistoryUtil()


def get_instance():
    match ncconfig.cf.save_type:
        case 'memory':
            return in_memory_util
        case 'redis':
            return redis_memory_util
