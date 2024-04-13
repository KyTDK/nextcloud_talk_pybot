
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
    def _save_to_memory(self, userid, history):
        pass

    @abstractmethod
    def _get_from_memory(self, userid):
        pass


    @abstractmethod
    def clear_memory(self, userid):
        pass
        
    def get_memory(self, userid):
        dict = self._get_from_memory(userid)
        if dict is None or not dict:  # Check for None and empty dictionary
            return ConversationBufferMemory(return_messages=True, chat_memory=ChatMessageHistory(messages=[]))
        memory_dict = self.__dict_to_message(dict)
        history = ChatMessageHistory()
        history.messages = history.messages + memory_dict
        return ConversationBufferMemory(return_messages=True, chat_memory=history)

    def get_base_memory(self, userid):
        dict = self._get_from_memory(userid)
        memory_dict = self.__dict_to_message(dict)
        return dict

    def save_memory(self, userid, history: ConversationBufferMemory):
        chat_memory = history.chat_memory
        memory = self.__tuncate_memory(chat_memory)
        self._save_to_memory(userid, memory)

    def count_tokens_in_dict(self, memory_dict, llm):
        tokens_in_history = 0
        for entry in memory_dict:
            if entry.get('data'):  # Check if 'data' key exists
                data = entry['data']
                if data.get('content'):  # Check if 'content' key exists within 'data'
                    content = data['content']
                    tokens_in_history+=llm.get_num_tokens(content)
        return tokens_in_history

    TOKEN_LIMIT = 1000

    def __tuncate_memory(self, history):
        #truncate conversation amount
        memory_dict = self.__message_to_dict(history)
        if len(memory_dict) > self.max_chat_history * 2:
            memory_dict = memory_dict[2:]
        #truncate token amount
        llm_gpt3 = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo-0125")
        tokens_in_history = self.count_tokens_in_dict(memory_dict, llm_gpt3)
        print("Tokens in history " + str(tokens_in_history))
        entry = memory_dict.pop(0)
        while tokens_in_history>self.TOKEN_LIMIT:
            if memory_dict:
                if entry.get('data'):  # Check if 'data' key exists
                    data = entry['data']
                    if data.get('content'):  # Check if 'content' key exists within 'data'
                        content = data['content']
                        if content:
                            content = content[:(tokens_in_history-self.TOKEN_LIMIT)]
                            entry['data']['content'] = content
                            print("Content: "+content)
                            tokens_in_history=self.count_tokens_in_dict(memory_dict, llm_gpt3)
                        else:
                            entry = memory_dict.pop(0)
        memory_dict.insert(0, entry)
        return memory_dict


    def _get_index_key(self, userid):
        return f'memory_{userid}'
    

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
