import ncbot.command.base as base

from ncbot.plugins.utils.history import get_instance
from ncbot.plugins.utils.custom_tools import ScrapeTool

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_community.utilities import SearxSearchWrapper


from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langchain_experimental.utilities import PythonREPL


from datetime import datetime

from ncbot.plugins.utils.custom_agent_maker import create_openai_tools_agent

plugin_name = 'openai'
model_gpt_3 = 'gpt-3.5-turbo-0125'

llm_gpt3 = ChatOpenAI(temperature=0.7, model_name=model_gpt_3)

reset=False

def set_reset(value):
  global reset  # Use `global` to access a variable from the enclosing scope
  reset = value


@base.command(plname=plugin_name, funcname='chat3', desc='Chat with Chatgpt using gpt-3.5-turbo model')
async def chat3(conversation_token, username, input):
    history_util = get_instance()
    history = history_util.get_memory(
        conversation_token).load_memory_variables({})['history']
    search = SearxSearchWrapper(searx_host="http://localhost:8888")
    wikipedia = WikipediaAPIWrapper()
    pubmed = PubMedAPIWrapper()
    python_repl = PythonREPL()
    scrape = ScrapeTool()
    tools = [
      Tool(
            name="Search",
            func=search.results,
            description="useful for when you need to answer questions about current events. You should ask targeted questions"
        ),
        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="useful when you need an answer about encyclopedic general knowledge"
        ),
        Tool(
            name='PubMed',
            func=pubmed.run, 
            description='Useful tool for querying medical publications'
        ),
        Tool(
            name="datetime",
            func=lambda x: datetime.now().isoformat(),
            description="Returns the current datetime"
        ),
        Tool(
            name="human_name",
            func=lambda x: username,
            description="Retrieve the name of the human, useful for greeting them and referring to them"
        ),
        Tool(
            name="python_repl",
            description="A Python shell, useful for solving complex tasks. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
            func=python_repl.run,
            ),
        Tool(
            name="forget",
            description="Clear AI's memory, forgets what everyone has said",
            func=lambda x: set_reset(True)
        ),
        scrape
    ]

    # Get the prompt to use - you can modify this!
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Your name is Nexty, you are capable of having both human-like conversations as well as being an assistant. You have access to real-time information as well as personal information of the user. When you don't know the answer to a question, you use the tools available to you to find an answer."),
            MessagesPlaceholder("history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
    # Construct the OpenAI Tools agent
    agent = create_openai_tools_agent(llm_gpt3, tools, prompt)
    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = await agent_executor.ainvoke({"input": input, "history": history}, verbose=True)
    if reset:
       history=[]
       set_reset(False)
    new_history = ConversationBufferMemory(
        return_messages=True, chat_memory=ChatMessageHistory(messages=history))
    new_history.save_context({"input": input}, {"output": response['output']})
    history_util.save_memory(conversation_token, new_history)
    return response['output']
