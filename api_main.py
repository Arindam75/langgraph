import json
import mlflow
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, BaseMessage
from typing import Dict, Any, AsyncGenerator
from workflow import build_app, close_app
from opentelemetry import trace

class QueryRequest(BaseModel):
    query: str
    thread_id: str = "1"
    actor_id: str = "andrew"


app = FastAPI()
langgraph_app = None
tracer = None

#mlflow.set_tracking_uri("http://127.0.0.1:5000")
#mlflow.set_experiment("langgraph-traces")
#mlflow.langchain.autolog()

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
    global langgraph_app, tracer
    tracer = trace.get_tracer("my-langgraph-service")
    langgraph_app = await build_app()
    print("app_initialized")


@app.on_event("shutdown")
async def shutdown_event():
    await close_app()
    print("memory_closed")


@app.post("/query")  # streaming -> do not rely on FastAPI response_model here
async def query_endpoint(request: QueryRequest):
    global langgraph_app, tracer
    config = {
        "recursion_limit": 10,
        # NOTE: LangGraph tracing/callbacks might already use OpenTelemetry
        # Check your LangGraph setup for an OTel callback handler.
        "configurable": {"thread_id": request.thread_id, "actor_id": request.actor_id}
    }

    inputs = {"messages": [HumanMessage(content=request.query)]}

    async def stream_responses() -> AsyncGenerator[bytes, None]:
        # Create a span for the async generator function
        with tracer.start_as_current_span("stream_responses_generator") as span_gen:
            try:
                # Create a span for the astream operation
                async for chunk in langgraph_app.astream(inputs, config, stream_mode="values"):
                    # Create a span for processing each chunk
                    with tracer.start_as_current_span("process_chunk") as span_chunk:
                        messages = chunk.get("messages") or []
                        last_message = messages[-1] if messages else None

                        # pull full state
                        state = await langgraph_app.aget_state(config)
                        state_vals = getattr(state, "values", {}) or {}

                        if state_vals.get("messages"): 
                            last_message = state_vals["messages"][-1]

                        if last_message is None:
                            span_chunk.set_attribute("event.skipped_chunk", True)
                            continue

                        payload = {
                            "answer": serialize_message(last_message),
                            "state": serialize_state(state_vals),
                        }
                        
                        # Add relevant data to the span
                        span_chunk.set_attribute("last_message_type", type(last_message).__name__)
                        span_chunk.set_attribute("output_payload_size", len(json.dumps(payload)))

                        yield (json.dumps(payload) + "\n").encode("utf-8")
                
                span_gen.set_attribute("stream.finished_successfully", True)

            except Exception as e:
                # Record exception on the overall generator span
                span_gen.record_exception(e)
                span_gen.set_status(trace.Status(trace.StatusCode.ERROR, f"Exception in stream: {e}"))
                yield (json.dumps({"error": str(e)}) + "\n").encode("utf-8")

    # Create the root span for the FastAPI endpoint
    with tracer.start_as_current_span("query_endpoint_request") as span_endpoint:
        span_endpoint.set_attribute("thread_id", request.thread_id)
        span_endpoint.set_attribute("actor_id", request.actor_id)
        # Note: The generator will inherit this context, allowing nested spans
        return StreamingResponse(stream_responses(), media_type="application/x-ndjson")