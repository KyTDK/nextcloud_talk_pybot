import ncbot.command.base as base

from ncbot.plugins.utils.history import get_instance
from ncbot.plugins.utils.custom_tools import ScrapeTool, SearchTool, FileGetByLocationTool, FileListTool

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, ChatMessageHistory


from langchain_experimental.utilities import PythonREPL


from datetime import datetime
import nc_py_api
import ncbot.config as ncconfig

from ncbot.plugins.utils.custom_agent_maker import create_tool_calling_agent

plugin_name = 'openai'
model_gpt_3 = 'gpt-4o-mini'

llm_gpt3 = ChatOpenAI(model_name=model_gpt_3, temperature=0.5)

reset=False

nc = nc_py_api.Nextcloud(nextcloud_url="http://cloud.neomechanical.com", nc_auth_user=ncconfig.cf.username, nc_auth_pass=ncconfig.cf.password)

def set_reset(value):
  global reset  # Use `global` to access a variable from the enclosing scope
  reset = value


@base.command(plname=plugin_name, funcname='chat3', desc='Chat with Chatgpt using gpt-3.5-turbo model')
async def chat3(conversation_token, username, user_id, input):
    history_util = get_instance()
    history = history_util.get_memory(
        conversation_token).load_memory_variables({})['history']
    python_repl = PythonREPL()
    scrape = ScrapeTool()
    search = SearchTool()

    get_files_by_location = FileGetByLocationTool()
    get_files_by_location.username = user_id
    get_files_by_location.nc = nc

    list_files = FileListTool()
    list_files.username = user_id
    list_files.nc = nc

    tools = [
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
        scrape,
        search,
        get_files_by_location,
        list_files
    ]

    # Get the prompt to use - you can modify this!
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant for the the Nextcloud server https://cloud.neomechanical.com. You know nothing, therefore you always research before answering a question, despite how cofident you think you are. If you browse the internet, you always provide your sources and urls to websites you accessed. You also have access to files the user shared, and can access them when the user asks."),
            MessagesPlaceholder("history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
    # Construct the OpenAI Tools agent
    agent = create_tool_calling_agent(llm_gpt3, tools, prompt)
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
