from typing_extensions import Literal
from typing import Any

import os
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
#from langchain_openai import ChatOpenAI
from langchain_aws.chat_models import ChatBedrock
from pydantic import Field
from bedrock_agentcore.memory import MemoryClient
from langgraph_checkpoint_aws import AgentCoreMemorySaver
from dotenv import load_dotenv
import asyncio

load_dotenv()

#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_ID = os.getenv("MODEL_ID")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
#model = ChatOpenAI(model=MODEL_ID, api_key=OPENAI_API_KEY)

model = ChatBedrock(
    model_id=MODEL_ID,
    region_name="ap-south-1",
    model_kwargs={
        "temperature": 0.7,
        # Other optional Bedrock parameters like 'topP', 'maxTokenCount' etc.
    },
)

memory_client = None
checkpointer = None


class MyState(MessagesState):
    summary: str = Field(default="")


def call_model(state: MyState):

    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]

    response = model.invoke(messages)
    return {"messages": response}


def summarize_conversation(state: MyState):

    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


def should_continue(state: MyState) -> Literal["summarize_conversation", END]:

    messages = state["messages"]
    if len(messages) > 3:
        return "summarize_conversation"
    return END


def build_workflow() -> StateGraph:

    # Define a new graph
    workflow = StateGraph(MyState)
    workflow.add_node("conversation", call_model)
    workflow.add_node(summarize_conversation)

    # Set the entrypoint as conversation
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)

    return workflow


async def build_app() -> Any:
    global memory_client, checkpointer

    if checkpointer is None:
        memory_name = "simple_chkpointer"
        memory_client = MemoryClient(region_name='ap-south-1')

        # Wrap sync call in executor to avoid blocking
        loop = asyncio.get_event_loop()
        memory = await loop.run_in_executor(None, memory_client.create_or_get_memory, memory_name)

        memory_id = memory["id"]
        print(f"Using memory id: {memory_id}")
        checkpointer = AgentCoreMemorySaver(memory_id, region_name='ap-south-1')

    workflow = build_workflow()
    return workflow.compile(checkpointer=checkpointer)


async def close_app():
    print("Shutting Down")
    global memory_client, checkpointer

    # No async teardown needed for AgentCoreMemorySaver
    memory_client = None
    checkpointer = None
