import json
import logging
import os
import traceback
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Dict
from uuid import UUID, uuid4
from enum import Enum
from datetime import datetime
from fastapi import BackgroundTasks
from threading import Lock
from copy import deepcopy
import time
import asyncio
from asyncio import Lock as AsyncLock

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langsmith import Client as LangsmithClient

from agents import DEFAULT_AGENT, get_agent, get_all_agent_info
from core import settings
from api_schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
    AgentStatus,
    AgentState,
)
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Construct agent with Sqlite checkpointer
    # TODO: It's probably dangerous to share the same checkpointer on multiple agents
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
        agents = get_all_agent_info()
        for a in agents:
            agent = get_agent(a.key)
            agent.checkpointer = saver
        yield
    # context manager will clean up the AsyncSqliteSaver on exit


app = FastAPI(lifespan=lifespan)
router = APIRouter(dependencies=[Depends(verify_bearer)])


@router.get("/info")
async def info() -> ServiceMetadata:
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )


def _parse_input(user_input: UserInput) -> tuple[dict[str, Any], UUID]:
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    kwargs = {
        "input": user_input.state if hasattr(user_input, 'state') else {"messages": [HumanMessage(content=user_input.message)]},
        "config": RunnableConfig(
            configurable={"thread_id": thread_id, "model": user_input.model,"max_concurrency": 10}, run_id=run_id
        ),
    }
    return kwargs, run_id


@router.post("/{agent_id}/invoke")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> Any:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    
    For state-based agents like marketing_agent, pass the initial state in the state field.
    For chat-based agents, pass the message in the message field.
    """
    agent: CompiledStateGraph = get_agent(agent_id)
    kwargs, run_id = _parse_input(user_input)
    try:
        response = await agent.ainvoke(**kwargs)
        
        # If response contains messages, format as ChatMessage
        if isinstance(response, dict) and "messages" in response:
            output = langchain_to_chat_message(response["messages"][-1])
            output.run_id = str(run_id)
            return output
            
        # Otherwise return the raw state
        return response
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    For state-based agents, it will stream state updates and final state.
    For chat-based agents, it streams messages and tokens.
    """
    agent: CompiledStateGraph = get_agent(agent_id)
    kwargs, run_id = _parse_input(user_input)

    #print("KWARGS", kwargs)

    def serialize_obj(obj):
        if hasattr(obj, 'model_dump'):  # Handle Pydantic models
            return obj.model_dump()
        elif isinstance(obj, (list, tuple)):  # Handle lists/tuples
            return [serialize_obj(item) for item in obj]
        elif isinstance(obj, dict):  # Handle dictionaries
            return {k: serialize_obj(v) for k, v in obj.items()}
        return str(obj)  # Fallback to string representation

    try:
        async for event in agent.astream(**kwargs, stream_mode="values"):
            print("EVENT", event)

            # Stream the event data
            if isinstance(event, dict):
                # Convert to JSON and yield as SSE data
                #print("EVENT IS DICT", event)
                yield f"data: {json.dumps(event, default=serialize_obj)}\n\n"
            else:
                # Convert non-dict events to string representation
                yield f"data: {str(event)}\n\n"

        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in message generator: {e}", exc_info=True)
        # Send error message to client
        error_msg = {"type": "error", "content": "An error occurred while processing your request"}
        yield f"data: {json.dumps(error_msg)}\n\n"
        yield "data: [DONE]\n\n"


def _sse_response_example() -> dict[int, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: {'type': 'state_update', 'content': {'personas': [...]}}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream", response_class=StreamingResponse, responses=_sse_response_example()
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    logger.info(f"Streaming response for user input: {user_input}")
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()


# TODO: Remove this or test it (currently not used)
@router.post("/history")
def history(input: ChatHistoryInput) -> ChatHistory:
    """
    Get chat history.
    """
    # TODO: Hard-coding DEFAULT_AGENT here is wonky
    agent: CompiledStateGraph = get_agent(DEFAULT_AGENT)
    try:
        state_snapshot = agent.get_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                }
            )
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


# Add these after existing imports
running_agents: Dict[str, AgentState] = {}
agents_lock = AsyncLock()

# Add these new endpoints
@router.post("/{agent_id}/start")
async def start_agent(
    background_tasks: BackgroundTasks,
    user_input: UserInput,
    agent_id: str = DEFAULT_AGENT
) -> dict:
    """Start an agent running in the background"""
    agent: CompiledStateGraph = get_agent(agent_id)
    kwargs, run_id = _parse_input(user_input)
    thread_id = kwargs["config"]["configurable"]["thread_id"]

    # Create new agent state tracker
    agent_state = AgentState()
    agent_state.thread_id = thread_id
    
    async with agents_lock:
        running_agents[str(run_id)] = agent_state
    
    async def run_agent():
        try:
            async for event in agent.astream(**kwargs, stream_mode="values"):
                # Create a new state update
                async with agents_lock:
                    agent_state.current_state = event
                    agent_state.last_update = datetime.utcnow()
            
            async with agents_lock:
                agent_state.status = AgentStatus.COMPLETED
        except Exception as e:
            logger.error(f"Agent error: {e}\nTraceback: {traceback.format_exc()}")
            async with agents_lock:
                agent_state.status = AgentStatus.ERROR
    
    # Run the agent
    background_tasks.add_task(run_agent)
    
    return {
        "run_id": str(run_id),
        "thread_id": thread_id,
        "status": "started"
    }

@router.get("/agent/{run_id}/status")
async def get_agent_status(run_id: str) -> dict:
    """Get the current status of a running agent"""
    start_time = time.time()
    print(f"\nStarting get_agent_status for run_id: {run_id}")
    
    # Take a quick snapshot of the state with the lock
    lock_start = time.time()
    async with agents_lock:
        if run_id not in running_agents:
            lock_end = time.time()
            print(f"Lock held for {(lock_end - lock_start)*1000:.2f}ms - Agent not found")
            raise HTTPException(
                status_code=404,
                detail="Agent not found. The run_id may be invalid or the agent has completed."
            )
        # Create a deep copy while holding the lock to prevent state changes during read
        agent_state = deepcopy(running_agents[run_id])
        lock_end = time.time()
        print(f"Lock held for {(lock_end - lock_start)*1000:.2f}ms")
    
    # Process the copied state without holding the lock
    process_start = time.time()
    response = {
        "run_id": run_id,
        "thread_id": agent_state.thread_id,
        "status": agent_state.status,
        "start_time": agent_state.start_time,
        "last_update": agent_state.last_update,
        "current_state": agent_state.current_state,
        "status_updates": agent_state.current_state.get("status_updates", []) if agent_state.current_state else []
    }
    process_end = time.time()
    print(f"Response processing took {(process_end - process_start)*1000:.2f}ms")
    
    end_time = time.time()
    print(f"Total get_agent_status time: {(end_time - start_time)*1000:.2f}ms\n")
    return response

# This is for browser use logs
@router.get("/logs")
async def list_logs() -> dict:
    """Get a list of available log files"""
    try:
        log_files = [f for f in os.listdir("logs") if f.endswith(".txt")]
        return {
            "log_files": log_files
        }
    except Exception as e:
        logger.error(f"Error listing log files: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving log files"
        )

@router.get("/logs/{filename}")
async def get_log_content(filename: str) -> dict:
    """Get the content of a specific log file"""
    try:
        file_path = os.path.join("logs", filename)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail="Log file not found"
            )
            
        if not filename.endswith(".txt"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type - only .txt files are supported"
            )
            
        with open(file_path, "r") as f:
            content = f.read()
            
        return {
            "filename": filename,
            "content": content
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading log file {filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error reading log file"
        )


app.include_router(router)
