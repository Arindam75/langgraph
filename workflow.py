from typing_extensions import Literal
from typing import Any

import os
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, RemoveMessage
from langchain_openai import ChatOpenAI
#from langgraph.checkpoint.memory import MemorySaver
from pydantic import Field
from langgraph.checkpoint.mysql.aio import AIOMySQLSaver
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_ID = os.getenv("MODEL_ID")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID")
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")


model = ChatOpenAI(model=MODEL_ID, api_key=OPENAI_API_KEY)
mysql_uri = f"mysql://{DB_USER}:{DB_PASSWORD}@localhost:3306/{DB_NAME}"
memory_cm = None
memory = None


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
    global memory_cm, memory
    if memory is None:
        # Create the async context manager
        memory_cm = AIOMySQLSaver.from_conn_string(mysql_uri)
        memory = await memory_cm.__aenter__()
        await memory.setup()

    workflow = build_workflow()
    return workflow.compile(checkpointer=memory)


async def close_app():
    print("Shutting Down")
    global memory_cm, memory
    if memory_cm is not None:
        await memory_cm.__aexit__(None, None, None)
        memory_cm = None
        memory = None