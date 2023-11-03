import os

import streamlit as st
from streamlit.logger import get_logger
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import (
    ConversationBufferMemory,
)
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.graphs import Neo4jGraph
from dotenv import load_dotenv
from utils import (
    create_vector_index,
)
from chains import (
    load_embedding_model,
    load_llm,
    configure_llm_only_chain,
    configure_qa_rag_chain,
)

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

# if Neo4j is local, you can go to http://localhost:7474/ to browse the database
neo4j_graph = Neo4jGraph(url=url, username=username, password=password)
embeddings, dimension = load_embedding_model(
    embedding_model_name,
    config={"ollama_base_url": ollama_base_url},
    logger=logger,
)
create_vector_index(neo4j_graph, dimension)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


llm = load_llm(
    llm_name, logger=logger, config={"ollama_base_url": ollama_base_url}
)

if "memory" not in st.session_state:
    MEMORY_KEY = "chat_history"
    st.session_state.memory = memory = AgentTokenBufferMemory(memory_key=MEMORY_KEY, llm=llm)
    # st.session_state.memory = ConversationBufferMemory(
    #     memory_key="chat_history", return_messages=True
    # )
# memory = ConversationSummaryBufferMemory(
#     llm=llm, input_key="question", output_key="answer"
# )
memory = st.session_state.memory
llm_chain = configure_llm_only_chain(llm)


if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = configure_qa_rag_chain(
        llm,
        embeddings,
        embeddings_store_url=url,
        username=username,
        password=password,
        memory=memory,
    )


rag_chain = st.session_state.rag_chain
#st.text(f"Chat history:{memory.dict()['chat_memory']['messages']}")


# Streamlit UI
styl = f"""
<style>
    /* not great support for :has yet (hello FireFox), but using it for now */
    .element-container:has([aria-label="Select RAG mode"]) {{
      position: fixed;
      bottom: 33px;
      background: white;
      z-index: 101;
    }}
    .stChatFloatingInputContainer {{
        bottom: 20px;
    }}

    /* Generate ticket text area */
    textarea[aria-label="Description"] {{
        height: 200px;
    }}

    .header-style {{
        font-size: 32px;
        font-weight: bold;
        color: #333;
        text-align: center;
    }}
</style>
"""
header_text = f"Clinical Study ChatBot"
st.markdown(f"<div class='header-style'>{header_text}</div>", unsafe_allow_html=True)
st.markdown(styl, unsafe_allow_html=True)


def chat_input():
    user_input = st.chat_input(
        "Please describe your patient for optimal matching..."
    )

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            #st.caption(f"RAG: enabled")
            stream_handler = StreamHandler(st.empty())
            result = rag_chain(
                {
                    "input": user_input,
                    # "chat_history": memory.dict()["chat_memory"]["messages"],
                },  # not sure wuth this empty chat history list
                callbacks=[stream_handler],
            )["output"]
            output = result
            st.session_state[f"user_input"].append(user_input)
            st.session_state[f"generated"].append(output)
            # st.session_state[f"rag_mode"].append(name)
            # print the chat history


def display_chat():
    # Session state
    if "generated" not in st.session_state:
        st.session_state[f"generated"] = []

    if "user_input" not in st.session_state:
        st.session_state[f"user_input"] = []

    if "rag_mode" not in st.session_state:
        st.session_state[f"rag_mode"] = []

    if st.session_state[f"generated"]:
        size = len(st.session_state[f"generated"])

        # Display only the last three exchanges
        for i in range(max(size - 3, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state[f"user_input"][i])

            with st.chat_message("assistant"):
                if i < len(
                    st.session_state[f"rag_mode"]
                ):  # Check to avoid IndexError
                    st.caption(f"RAG: {st.session_state[f'rag_mode'][i]}")
                else:
                    st.caption("RAG: Unknown Mode")
                st.write(st.session_state[f"generated"][i])

        # with st.expander("Not finding what you're looking for?"):
        #     st.write(
        #         "Automatically generate a draft for an internal ticket to our support team."
        #     )
        #     st.button(
        #         "Generate ticket",
        #         type="primary",
        #         key="show_ticket",
        #         on_click=open_sidebar,
        #     )
        # with st.container():
        #     st.write("&nbsp;")

display_chat()
chat_input()
