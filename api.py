import os
import json
from langchain.graphs import Neo4jGraph
from dotenv import load_dotenv
from utils import (
    create_vector_index,
    BaseLogger,
)
from chains import (
    load_embedding_model,
    load_llm,
    configure_llm_only_chain,
    configure_qa_rag_chain,
)
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from langchain.callbacks.base import BaseCallbackHandler
from threading import Thread
from queue import Queue, Empty
from collections.abc import Generator
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
os.environ["NEO4J_URL"] = url

embeddings, dimension = load_embedding_model(
    embedding_model_name,
    config={ollama_base_url: ollama_base_url},
    logger=BaseLogger(),
)

neo4j_graph = Neo4jGraph(url=url, username=username, password=password)
create_vector_index(neo4j_graph, dimension)

llm = load_llm(
    llm_name, logger=BaseLogger(), config={"ollama_base_url": ollama_base_url}
)

llm_chain = configure_llm_only_chain(llm)
rag_chain = configure_qa_rag_chain(
    llm,
    embeddings,
    embeddings_store_url=url,
    username=username,
    password=password,
)


class QueueCallback(BaseCallbackHandler):
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

chat_memory = {}


@app.get("/")
async def root():
    return {"message": "Hello World"}


class Question(BaseModel):
    text: str
    rag: bool = False
    session_id: str = None


@app.get("/query-stream")
def qstream(question: Question = Depends()):
    session_id = question.session_id
    chat_history = chat_memory.get(session_id, [])

    output_function = llm_chain
    if question.rag:
        output_function = rag_chain

    q = Queue()

    def cb():
        output_function(
            {"question": question.text, "chat_history": chat_history},
            callbacks=[QueueCallback(q)],
        )

    def generate():
        yield json.dumps({"init": True, "model": llm_name})
        for token, _ in stream(cb, q):
            yield json.dumps({"token": token})

    return EventSourceResponse(generate(), media_type="text/event-stream")


@app.get("/query")
async def ask(question: Question = Depends()):
    session_id = question.session_id
    chat_history = chat_memory.get(session_id, [])
    print("test1:", chat_memory)

    output_function = llm_chain
    if question.rag:
        output_function = rag_chain

    result = output_function(
        {"question": question.text, "chat_history": chat_history}, callbacks=[]
    )

    chat_memory[session_id] = chat_history + [
        {"user": question.text, "assistant": result["answer"]}
    ]
    # print("test2:"chat_memory)

    return json.dumps({"result": result["answer"], "model": llm_name})
