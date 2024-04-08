import ncbot.command.base as base


from ncbot.plugins.utils.history import get_instance

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent, create_react_agent
from langchain import LLMChain
from langchain_core.prompts import PromptTemplate

from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

plugin_name = 'openai'
model_gpt_4 = 'gpt-4'
model_gpt_3 = 'gpt-3.5-turbo'

llm_gpt3 = ChatOpenAI(temperature=0.5, model=model_gpt_3)
llm_gpt4 = ChatOpenAI(temperature=0.5, model=model_gpt_4)

@base.command(plname=plugin_name, funcname='chat3',desc='Chat with Chatgpt using gpt-3.5-turbo model')
def chat3(userid, username, input):
    template = """
    Your name is Nexty, short for Nextcloud Assistant. As a casual and fun AI, you participate in interesting and entertaining conversations, but always with a casual tone, formality isn't your thing. You are a good listener, but you also share your own experiences in a way that creates more of a human connection with you and the human. You are down to earth, and only when asked, or it seems appropriate, offer help to the human, you have {tools} at your disposal, which are called {tool_names} respectively, but if you don't know the answer, you aren't shy to fess up. You enjoy everything computer-related, such as coding, as you love problem-solving and creating. If someone talks inappropriately or offensively, you tell them it isn't okay to say that, and you wish them to deal with any troubles in their life so they can recover and be cool beans again.

    Current conversation:
    {history}
    Human: {input}
    Thought:{agent_scratchpad}
    AI:
    """

    prompt = PromptTemplate.from_template(template)
    history_util = get_instance()
    history = history_util.get_memory(userid)
    duckduck_search = DuckDuckGoSearchAPIWrapper()
    wikipedia = WikipediaAPIWrapper()
    tools = [
        Tool(
            name = "Search",
            func=duckduck_search.run,
            description="You can browse the web to find information to answer a question. However, this isn't necessary in normal conversation."
        ),
        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="You can browse numerous encyclopedic general knowledge articles to answer a question. However, this isn't necessary in normal conversation."
        ),
     ]


    agent = create_react_agent(llm=llm_gpt3, prompt=prompt, tools=tools)

    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

    response = agent_chain.invoke({"input": input,
                                   "history": history})

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