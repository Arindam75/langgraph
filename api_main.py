import json
import asyncio
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, BaseMessage
from typing import Dict, Any, AsyncGenerator
from workflow import build_app, close_app


class QueryRequest(BaseModel):
    query: str
    thread_id: str = "1"


app = FastAPI()
langgraph_app = None


def serialize_message(msg: BaseMessage):
    return {
        "type": msg.type,
        "content": msg.content,
        "additional_kwargs": getattr(msg, "additional_kwargs", {}),
        "example": getattr(msg, "example", False),
    }


def serialize_state(state: dict):
    out = {}
    for k, v in state.items():
        if isinstance(v, list) and all(isinstance(m, BaseMessage) for m in v):
            out[k] = [serialize_message(m) for m in v]
        elif isinstance(v, BaseMessage):
            out[k] = serialize_message(v)
        else:
            # e.g. summary: str
            out[k] = v
    return out


@app.on_event("startup")
async def startup_event():
    global langgraph_app
    langgraph_app = await build_app()
    print("app_initialized")


@app.on_event("shutdown")
async def shutdown_event():
    await close_app()
    print("memory_closed")


@app.post("/query")  # streaming -> do not rely on FastAPI response_model here
async def query_endpoint(request: QueryRequest):
    config = {
        "recursion_limit": 10,
        "configurable": {"thread_id": request.thread_id}
    }

    # IMPORTANT: use the key your graph expects (commonly "messages", not "message")
    inputs = {"messages": [HumanMessage(content=request.query)]}

    async def stream_responses() -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in langgraph_app.astream(inputs, config, stream_mode="values"):
                messages = chunk.get("messages") or []
                last_message = messages[-1] if messages else None

                # pull full state
                state = await langgraph_app.aget_state(config)
                state_vals = getattr(state, "values", {}) or {}

                if state_vals.get("messages"): 
                    last_message = state_vals["messages"][-1]

                if last_message is None:
                    continue

                payload = {
                    "answer": serialize_message(last_message),
                    "state": serialize_state(state_vals),
                }

                yield (json.dumps(payload) + "\n").encode("utf-8")
        except Exception as e:
            yield (json.dumps({"error": str(e)}) + "\n").encode("utf-8")

    return StreamingResponse(stream_responses(), media_type="application/x-ndjson")
