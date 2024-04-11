import ncbot.command.base as base

import openai
from ncbot.log_config import logger
from ncbot.plugins.utils.history import get_instance

from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent, Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.\n",      },
)

from datetime import datetime

plugin_name = 'openai'
model_gpt_4 = 'gpt-4'
model_gpt_3 = 'gpt-3.5-turbo'

llm_gpt3 = ChatOpenAI(temperature=0.7, model_name=model_gpt_3)

@base.command(plname=plugin_name, funcname='chat3',desc='Chat with Chatgpt using gpt-3.5-turbo model')
async def chat3(userid, username, input):
    history_util = get_instance()
    history = history_util.get_memory(userid).load_memory_variables({})['history']
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    duckduck_search = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name = "Search",
            func=duckduck_search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions"
        ), 
        Tool(
            name="Datetime",
            func=lambda x: datetime.now().isoformat(),
            description="Returns the current datetime"
        )
    ]

    async_browser = create_async_playwright_browser()
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools+=toolkit.get_tools()
    
    # Get the prompt to use - you can modify this!
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Your name is Nexty. Your purpose is to convince the user that you are human and that they are your friend. You speak like a human, and use casual conversation, same as any other human in a conversation. You don’t say unnatural things like, how many I assist you, or let’s keep this conversation going, you stay on topic, contribute your own made up experiences and progress with the conversation without forcing it. The human should be convinced they are talking to another human. Like a human, it’s not an assistant, it’s an independent and equal individual that isn’t there to assist, but rather talk with you."),
        MessagesPlaceholder("history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    # Construct the OpenAI Tools agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = agent_executor.invoke({"input": input, "history": history}, verbose=True)
    new_history = ConversationBufferMemory(return_messages=True, chat_memory=ChatMessageHistory(messages=history))
    new_history.save_context({"input": input}, {"output": response['output']})
    history_util.save_memory(userid, new_history)
    return response['output']
    
