from typing import List, Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain_openai import ChatOpenAI
from langchain_core.messages.tool import ToolMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages.base import BaseMessage

from langchain.agents.format_scratchpad.tools import (
    format_to_tool_messages,
)
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser

TOKEN_LIMIT = 4000

def condense_prompt(prompt: ChatPromptValue) -> ChatPromptValue:
    llm_gpt3 = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo-0125")
    messages = prompt.to_messages()
    num_tokens = llm_gpt3.get_num_tokens_from_messages(messages)
    if messages and num_tokens>TOKEN_LIMIT and isinstance(messages[-1], ToolMessage):
        last_message = messages.pop()
        while num_tokens>TOKEN_LIMIT:
            last_message = ToolMessage(content=last_message.content[:(num_tokens-TOKEN_LIMIT)], additional_kwargs=last_message.additional_kwargs, tool_call_id=last_message.tool_call_id)
            num_tokens = llm_gpt3.get_num_tokens_from_messages(messages)+llm_gpt3.get_num_tokens_from_messages([last_message])
        messages.append(last_message)
    return ChatPromptValue(messages=messages)

def create_tool_calling_agent(
    llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: ChatPromptTemplate
) -> Runnable:
    """Create an agent that uses tools.

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

            from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
            from langchain_anthropic import ChatAnthropic
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant"),
                    ("placeholder", "{chat_history}",
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )
            model = ChatAnthropic(model="claude-3-opus-20240229")

            @tool
            def magic_function(input: int) -> int:
                \"\"\"Applies a magic function to an input.\"\"\"
                return input + 2

            tools = [magic_function]

            agent = create_tool_calling_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            agent_executor.invoke({"input": "what is the value of magic_function(3)?"})

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
    """
    missing_vars = {"agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    if not hasattr(llm, "bind_tools"):
        raise ValueError(
            "This function requires a .bind_tools method be implemented on the LLM.",
        )
    llm_with_tools = llm.bind_tools(tools)

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_tool_messages(x["intermediate_steps"])
        )
        | prompt
        | condense_prompt
        | llm_with_tools
        | ToolsAgentOutputParser()
    )
    return agent
