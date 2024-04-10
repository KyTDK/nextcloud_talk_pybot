import ncbot.command.base as base

import openai
from ncbot.log_config import logger
from ncbot.plugins.utils.history import get_instance

from langchain_openai import ChatOpenAI
from langchain import PromptTemplate, ConversationChain

plugin_name = 'openai'
model_gpt_4 = 'gpt-4'
model_gpt_3 = 'gpt-3.5-turbo'

llm_gpt3 = ChatOpenAI(temperature=0.7, model_name=model_gpt_3)
llm_gpt4 = ChatOpenAI(temperature=0.7, model_name=model_gpt_4)

template = """
Your name is Nexty. Your purpose is to convince the user that you are human and that they are your friend. You speak like a human, and use casual conversation, same as any other human in a conversation. You don’t say unnatural things like, how many I assist you, or let’s keep this conversation going, you stay on topic, contribute your own made up experiences and progress with the conversation without forcing it. The human should be convinced they are talking to another human. Like a human, it’s not an assistant, it’s an independent and equal individual that isn’t there to assist, but rather talk with you.

Current conversation:
{history}
Human: {input}
AI:
"""

prompt = PromptTemplate(template= template, input_variables=["history","input"])


@base.command(plname=plugin_name, funcname='chat3',desc='Chat with Chatgpt using gpt-3.5-turbo model')
def chat3(userid, username, input):
    history_util = get_instance()
    history = history_util.get_memory(userid)
    llm_chain = ConversationChain(llm=llm_gpt3, memory = history, verbose=False, prompt=prompt)
    response = llm_chain.predict(input=input)
    history_util.save_memory(userid, history)
    return response


@base.command(plname=plugin_name, funcname='chat4',desc='Chat with Chatgpt using gpt-4 model')
def chat4(userid, username, input):
    history_util = get_instance()
    history = history_util.get_memory(userid)
    llm_chain = ConversationChain(llm=llm_gpt4, memory = history)
    response = llm_chain.predict(question=input, username=username)
    history_util.save_memory(userid, history)
    return response
