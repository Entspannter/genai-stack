import os

from langchain.graphs import Neo4jGraph
from dotenv import load_dotenv
from utils import (
    create_vector_index,
    BaseLogger,
    Neo4jChatMessageHistory,
)
from chains import (
    load_embedding_model,
    load_llm,
    configure_llm_only_chain,
    configure_qa_rag_chain,
)
from fastapi import FastAPI, Depends, Cookie, HTTPException
from pydantic import BaseModel
from langchain.callbacks.base import BaseCallbackHandler
from threading import Thread
from queue import Queue, Empty
from collections.abc import Generator
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from fastapi import FastAPI, Response, Request, Cookie

print("starting api.py")
load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

embeddings, dimension = load_embedding_model(
    embedding_model_name,
    config={ollama_base_url: ollama_base_url},
    logger=BaseLogger(),
)

# if Neo4j is local, you can go to http://localhost:7474/ to browse the database
neo4j_graph = Neo4jGraph(url=url, username=username, password=password)
create_vector_index(neo4j_graph, dimension)

llm = load_llm(
    llm_name, logger=BaseLogger(), config={"ollama_base_url": ollama_base_url}
)


class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs) -> None:
        return self.q.empty()


def stream(cb, q) -> Generator:
    job_done = object()

    def task():
        x = cb()
        q.put(job_done)

    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            content += next_token
            yield next_token, content
        except Empty:
            continue


app = FastAPI()
origins = ["http://localhost:8505", "http://127.0.0.1:8505"]  # ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


class Question(BaseModel):
    text: str
    rag: bool = False


class BaseTicket(BaseModel):
    text: str


@app.get("/query-stream")
async def qstream(session_id: str, question: Question = Depends()):
    # Use session_id to fetch the relevant chat history or data from the database
    message_history = Neo4jChatMessageHistory(session_id, url, username, password)
    MEMORY_KEY = "chat_history"
    memory = AgentTokenBufferMemory(
        memory_key=MEMORY_KEY, llm=llm, chat_memory=message_history
    )

    # Configure the QA RAG chain or default output function
    rag_chain = configure_qa_rag_chain(
        llm,
        embeddings,
        embeddings_store_url=url,
        username=username,
        password=password,
        memory=memory,
    )
    output_function = llm_chain if not question.rag else rag_chain

    q = Queue()

    def cb():
        # Pass the necessary data to the output function
        output_function(
            {"input": question.text, "chat_history": memory.chat_memory.messages},
            callbacks=[QueueCallback(q)],
        )

    def generate():
        # Initialize the stream
        yield json.dumps({"init": True, "model": llm_name})
        # Stream the data
        for token, _ in stream(cb, q):
            yield json.dumps({"token": token})
        yield json.dumps({"end": True})

    # Return an event source response for SSE
    return EventSourceResponse(generate(), media_type="text/event-stream")


@app.get("/query")
async def ask(
    response: Response,
    request: Request,
    question: Question = Depends(),
    chat_history: list = Depends(),
):
    # Extract session_id from cookie
    session_id_from_cookie = request.cookies.get("session_id")

    # Get or create session and chat history
    session_id, memory = get_or_create_session(
        session_id=session_id_from_cookie,
        llm=llm,  # Assuming llm is defined elsewhere in your app
        neo4j_graph=neo4j_graph,  # Assuming neo4j_graph is your Neo4j connection
        url=url,  # Neo4j URL
        username=username,  # Neo4j username
        password=password,  # Neo4j password
    )
    print("chat_history /query", memory.chat_memory.messages)
    print("session_id /query", session_id)
    if session_id_from_cookie != session_id:
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age=1800,
            path="/",
            samesite="lax",
        )

    # Configure the QA RAG chain or use the default output function
    rag_chain = configure_qa_rag_chain(
        llm,
        embeddings,
        embeddings_store_url=url,
        username=username,
        password=password,
        memory=memory,
    )
    output_function = llm_chain if not question.rag else rag_chain

    # Append the current question to the chat history
    memory.chat_memory.messages.append(question.text)

    # Execute the output function
    result = output_function(
        {"question": question.text, "chat_history": memory.chat_memory.messages},
        callbacks=[],
    )

    # Update the session with the new chat history in Neo4J
    neo4j_graph.query(
        "MATCH (s:Session {id: $session_id}) SET s.chat_history = $chat_history",
        params={
            "session_id": session_id,
            "chat_history": json.dumps(memory.chat_memory.messages),
        },
    )

    # Return the result
    return {"result": result["output"], "model": llm_name, "session_id": session_id}


def get_or_create_session(session_id: str, llm, neo4j_graph, url, username, password):
    """
    Retrieve or create a session in Neo4J. If no session_id is provided, a new session
    is created. The function connects to a Neo4J database to record the session.

    :param session_id: The current session ID, if available.
    :param llm: A parameter specific to your application's logic.
    :param neo4j_graph: The Neo4j graph object for database interaction.
    :param url: Neo4j database URL.
    :param username: Neo4j database username.
    :param password: Neo4j database password.
    :return: A tuple containing the session_id and a memory object.
    """
    # Create a new session if no session_id is provided
    if session_id is None:
        session_id = str(uuid.uuid4())
        # Insert the new session into Neo4j database
        neo4j_graph.query(
            "CREATE (s:Session {id: $session_id, chat_history: $chat_history})",
            params={"session_id": session_id, "chat_history": json.dumps([])},
        )

    # Common logic for both new and existing sessions
    message_history = Neo4jChatMessageHistory(session_id, url, username, password)
    MEMORY_KEY = "chat_history"
    memory = AgentTokenBufferMemory(
        memory_key=MEMORY_KEY, llm=llm, chat_memory=message_history
    )

    return session_id, memory


@app.get("/manage-session")
async def manage_session(response: Response, request: Request):
    # Extract session_id from cookie
    session_id_from_cookie = request.cookies.get("session_id")
    print("session_id_from_cookie", session_id_from_cookie)

    # Get or create session and chat history
    session_id, memory = get_or_create_session(
        session_id=session_id_from_cookie,
        llm=llm,  # Assuming llm is defined elsewhere in your app
        neo4j_graph=neo4j_graph,  # Neo4j connection
        url=url,  # Neo4j URL
        username=username,  # Neo4j username
        password=password,  # Neo4j password
    )
    print("new session_id", session_id)

    # Set or update the session_id cookie
    if session_id_from_cookie != session_id:
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age=1800,
            path="/",
            samesite="lax",
        )

    # Return some response, e.g., confirmation or the session ID
    return {"session_id": session_id}


@app.post("/reset-session")
async def reset_session(response: Response):
    # Reset the session_id cookie
    response.delete_cookie(key="session_id")
    return {"message": "Session reset"}


# session_id, memory = get_or_create_session()

llm_chain = configure_llm_only_chain(llm)
