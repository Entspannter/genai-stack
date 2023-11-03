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
from fastapi import FastAPI, Response, Request

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

def set_session_id(response: Response, value: str):
    """Helper function to set a cookie."""
    response.set_cookie(key="session_id", value=value)

def get_session_id(request: Request) -> str:
    """Helper function to retrieve a cookie."""
    return request.cookies.get("session_id", None)

def get_or_create_session(session_id: str = Cookie(None), llm=llm):
    """
    Retrieve or create a session in Neo4J.
    """
    session_id = get_session_id(request = Request)
    if session_id is None:
        # Create new session
        session_id = str(uuid.uuid4())
        set_session_id(response= Response, value = session_id)
        # TODO: connect to user name and time stamp (to make it very!!! unique)
        neo4j_graph.run("CREATE (s:Session {id: $session_id, chat_history: $chat_history})", session_id=session_id, chat_history=json.dumps([]))
        message_history = Neo4jChatMessageHistory(session_id, url, username, password )
        MEMORY_KEY = "chat_history"
        memory = AgentTokenBufferMemory(memory_key=MEMORY_KEY, llm=llm, chat_memory=chat_history)
    else:
        message_history = Neo4jChatMessageHistory(session_id, url, username, password )

        MEMORY_KEY = "chat_history"
        memory = AgentTokenBufferMemory(memory_key=MEMORY_KEY, llm=llm, chat_memory=message_history)

    return session_id, memory


session_id, memory = get_or_create_session()
llm_chain = configure_llm_only_chain(llm)
rag_chain = configure_qa_rag_chain(
    llm, embeddings, embeddings_store_url=url, username=username, password=password, memory= memory
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
origins = ["*"]

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
def qstream(question: Question = Depends()):
    output_function = llm_chain
    if question.rag:
        output_function = rag_chain

    q = Queue()

    def cb():
        output_function(
            {"question": question.text, "chat_history": []},
            callbacks=[QueueCallback(q)],
        )

    def generate():
        yield json.dumps({"init": True, "model": llm_name})
        for token, _ in stream(cb, q):
            yield json.dumps({"token": token})

    return EventSourceResponse(generate(), media_type="text/event-stream")


@app.get("/query")
async def ask(question: Question = Depends(), chat_history: list = Depends(get_or_create_session)):
    output_function = llm_chain
    if question.rag:
        output_function = rag_chain
    
    chat_history[0].append(question.text)
    result = output_function(
        {"question": question.text, "chat_history": chat_history[0]}, callbacks=[]
    )

    # Update the session with the new chat history in Neo4J
    neo4j_graph.run("MATCH (s:Session {id: $session_id}) SET s.chat_history = $chat_history", session_id=chat_history[1], chat_history=json.dumps(chat_history[0]))

    return {"result": result["output"], "model": llm_name, "session_id": chat_history[1]}