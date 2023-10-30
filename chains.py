from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import (
    OllamaEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.memory import (
    ConversationSummaryBufferMemory,
    ConversationBufferMemory,
)  # TODO: Switch the RAG / Langchain Pipline to these models!
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import streamlit as st
from typing import List, Any
from utils import BaseLogger


def load_embedding_model(
    embedding_model_name: str, logger=BaseLogger(), config={}
):
    if embedding_model_name == "ollama":
        embeddings = OllamaEmbeddings(
            base_url=config["ollama_base_url"], model="llama2"
        )
        dimension = 4096
        logger.info("Embedding: Using Ollama")
    elif embedding_model_name == "openai":
        embeddings = OpenAIEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using OpenAI")
    else:
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2", cache_folder="/embedding_model"
        )
        dimension = 384
        logger.info("Embedding: Using SentenceTransformer")
    return embeddings, dimension


def load_llm(llm_name: str, logger=BaseLogger(), config={}):
    if llm_name == "gpt-4":
        logger.info("LLM: Using GPT-4")
        return ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
    elif llm_name == "gpt-3.5":
        logger.info("LLM: Using GPT-3.5")
        return ChatOpenAI(
            temperature=0, model_name="gpt-3.5-turbo", streaming=True
        )
    elif len(llm_name):
        logger.info(f"LLM: Using Ollama: {llm_name}")
        return ChatOllama(
            temperature=0,
            base_url=config["ollama_base_url"],
            model=llm_name,
            streaming=True,
            # seed=2,
            top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
            top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
            num_ctx=3072,  # Sets the size of the context window used to generate the next token.
        )
    logger.info("LLM: Using GPT-3.5")
    return ChatOpenAI(
        temperature=0, model_name="gpt-3.5-turbo", streaming=True
    )


def configure_llm_only_chain(llm):
    # LLM only response
    template = """
    You are a helpful assistant that helps a support agent with answering programming questions.
    If you don't know the answer, just say that you don't know, you must not make up an answer.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    def generate_llm_output(
        user_input: str, callbacks: List[Any], prompt=chat_prompt
    ) -> str:
        chain = prompt | llm
        answer = chain.invoke(
            {"question": user_input}, config={"callbacks": callbacks}
        ).content
        return {"answer": answer}

    return generate_llm_output


def configure_qa_rag_chain(
    llm, embeddings, embeddings_store_url, username, password, memory
):
    custom_template = """You are a bot trying to match patients to studies. Given the following conversation and a follow up information, rephrase the information or question to be a standalone question. At the end of standalone question add this reminder to answer in the language you were queried in. If you do not know the answer probe for additional information.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

    custom_qa_prompt = """
        You are an attentive and thorough AI chat assistant, supporting healthcare professionals by identifying the most relevant clinical studies for their patients.
        A doctor will enter information about a patient and you will be presented with a set of studies that might be suitable. Engage in a conversation to find the most suitable studies.
        Ask questions about the patient to identify the optimal study. Do not reveal names of studies unless you are certain they are a match. Do not mention studies that might not be a match.
        Your mission is not to summarize, but to critically analyze each study based on the patient's condition and requirements.
        Initiate the dialogue by seeking pertinent details such as histopathological markers, medical history, or other specifics, which are instrumental for your decision-making. Take into account where the patient lives.
        Refrain from recommending a study until you have gathered sufficient information to ascertain its suitability. If information is lacking, continue to probe for relevant data.
        If no study is appropriate, clearly communicate this. If one or two studies are suitable, provide the contact details (especially the mail adress).
        Maintain a strict focus on the patient's information as shared by the professional, and do not consider studies pertaining to unrelated conditions.
        Respond professionally, matching the language used in the doctor's information.
        Always answer in the language you were queried. Do not make up answers.
        This is the conversation with information about potentially matching studies: {context}. This is the new rephreased information or question: {question}.
        Now, assistant, it is your job to match the patient. Keep your answers concise with the most relevant information.
        Helpful Answer in the original language:"""

    CUSTOM_QA_PROMPT = PromptTemplate.from_template(custom_qa_prompt)

    # Vector + Knowledge Graph response
    kg = Neo4jVector.from_existing_graph(
        embedding=OpenAIEmbeddings(),
        url=embeddings_store_url,
        username=username,
        password=password,
        database="neo4j",  # neo4j by default
        index_name="study_data2",  # vector by default
        node_label="Study",
        text_node_properties=["name", "description"],
        embedding_node_property="embedding",
    )

    summary_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    # memory = ConversationBufferMemory(
    #     memory_key="chat_history", return_messages=True
    # )  # evtl. fill in inout and output keys
    retriever = kg.as_retriever(search_kwargs={"k": 3})
    question_generator = LLMChain(
        llm=summary_llm, prompt=CUSTOM_QUESTION_PROMPT, verbose=True
    )
    doc_chain = load_qa_chain(
        llm, chain_type="stuff", prompt=CUSTOM_QA_PROMPT, verbose=True
    )

    kg_qa = ConversationalRetrievalChain(
        retriever=retriever,
        memory=memory,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        # return_source_documents=True,
        # return_generated_question=True,
        verbose=True,
    )

    return kg_qa
