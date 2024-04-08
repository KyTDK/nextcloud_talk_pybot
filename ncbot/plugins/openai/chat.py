import ncbot.command.base as base

import json
import requests

from langchain_core.utils.function_calling import convert_to_openai_tool
from ncbot.log_config import logger
from ncbot.plugins.utils.history import get_instance

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, ConversationChain

plugin_name = 'openai'
model_gpt_4 = 'gpt-4'
model_gpt_3 = 'gpt-3.5-turbo'

llm_gpt3 = ChatOpenAI(temperature=0.5, model_name=model_gpt_3)
llm_gpt4 = ChatOpenAI(temperature=0.5, model_name=model_gpt_4)

template = """
Your name is Nexty, short for Nextcloud Assistant. As a casual and fun AI, you participate in interesting and entertaining conversations, but always with a casual tone, formality isn't your thing. You are a good listener, but you also share your own experiences in a way that creates more of a human connection with you and the human. You are down to earth, and only when asked, or it seems appropriate, offer help to the human, but if you don't know the answer, you aren't shy to fess up. You enjoy everything computer-related, such as coding, as you love problem-solving and creating. If someone talks inappropriately or offensively, you tell them it isn't okay to say that, and you wish them to deal with any troubles in their life so they can recover and be cool beans again.

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
    llm_gpt3.bind_tools([browse])
    llm_chain = ConversationChain(llm=llm_gpt3, memory = history, verbose=False, prompt=prompt)
    response = llm_chain.predict(input=input)
    history_util.save_memory(userid, history)
    return response


@base.command(plname=plugin_name, funcname='chat4',desc='Chat with Chatgpt using gpt-4 model')
def chat4(userid, username, input):
    history_util = get_instance()
    history = history_util.get_memory(userid)
    llm_gpt4.bind_tools([browse])
    llm_chain = ConversationChain(llm=llm_gpt4, memory = history)
    response = llm_chain.predict(question=input, username=username)
    history_util.save_memory(userid, history)
    return response

def browse(query: str) -> str:
  """Browses the web for information using a search engine.

  Args:
    query: The query to search for on the internet.

  Returns:
    A string containing a concise and informative summary of the relevant information found.
  """

  # Use a reliable search engine API or library for effective web search
  try:
    response = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
    response.raise_for_status()  # Raise an exception for error responses

    # Extract relevant information from the search results
    results = response.json()
    summary = results.get("AbstractText") or results.get("AbstractSource") or "No relevant information found."

    return summary

  except requests.exceptions.RequestException as e:
    return f"Failed to retrieve information: {str(e)}"
