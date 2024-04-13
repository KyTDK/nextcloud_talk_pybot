from typing import List, Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.prompt_values import ChatPromptValue
from langchain_openai import ChatOpenAI
from langchain_core.messages.base import BaseMessage
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_core.messages.tool import ToolMessage
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

def get_tool_messages(messages: List[BaseMessage]) -> List[ToolMessage]:
  """Extracts all BaseMessage objects of type ToolMessage from the messages list.

  Args:
      messages: A list of BaseMessage objects.

  Returns:
      A list containing only objects of type ToolMessage.
  """
  tool_messages = []
  for message in messages:
    if isinstance(message, ToolMessage):
      tool_messages.append(message)
  return tool_messages

def condense_prompt(prompt: ChatPromptValue) -> ChatPromptValue:
    llm_gpt3 = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo-0125")
    messages = prompt.to_messages()
    num_tokens = llm_gpt3.get_num_tokens_from_messages(messages)
    tool_messages = get_tool_messages(messages)
    if tool_messages:
        last_tool_message = tool_messages.pop()
        new_last_tool_message = last_tool_message
        while num_tokens>4000:
            new_last_tool_message = ToolMessage(content=last_tool_message.content[:1], additional_kwargs=last_tool_message.additional_kwargs, tool_call_id=last_tool_message.tool_call_id)
        #replace old tool message with new, truncated one
        for i, n in enumerate(messages):
        if n == last_tool_message:
            messages[i] = new_last_tool_message
    return ChatPromptValue(messages=messages)

def create_openai_tools_agent(
    llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: ChatPromptTemplate
) -> Runnable:
    """Create an agent that uses OpenAI tools.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use. See Prompt section below for more on the expected
            input variables.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Example:

        .. code-block:: python

            from langchain import hub
            from langchain_community.chat_models import ChatOpenAI
            from langchain.agents import AgentExecutor, create_openai_tools_agent

            prompt = hub.pull("hwchase17/openai-tools-agent")
            model = ChatOpenAI()
            tools = ...

            agent = create_openai_tools_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            agent_executor.invoke({"input": "hi"})

            # Using with chat history
            from langchain_core.messages import AIMessage, HumanMessage
            agent_executor.invoke(
                {
                    "input": "what's my name?",
                    "chat_history": [
                        HumanMessage(content="hi! my name is bob"),
                        AIMessage(content="Hello Bob! How can I assist you today?"),
                    ],
                }
            )

    Prompt:

        The agent prompt must have an `agent_scratchpad` key that is a
            ``MessagesPlaceholder``. Intermediate agent actions and tool output
            messages will be passed in here.

        Here's an example:

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant"),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            )
    """
    missing_vars = {"agent_scratchpad"}.difference(prompt.input_variables)
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    llm_with_tools = llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            )
        )
        | prompt
        | condense_prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    return agent
