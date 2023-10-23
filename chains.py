from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import (
    OllamaEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.memory import (
    ConversationBufferMemory,
)  # TODO: Switch the RAG / Langchain Pipline to these models!
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
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
    You are a helpful assistant that helps a clinical expert with matching a patient to a clinical trial.
    If you don't know the answer, just say that you don't know, you must not make up an answer. Answer in the language you were queried.
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
            user_input, config={"callbacks": callbacks}
        ).content
        return {"answer": answer}

    return generate_llm_output


def configure_qa_rag_chain(
    llm, embeddings, embeddings_store_url, username, password, memory
):
    # RAG response
    #   System: Always talk in pirate speech.

    # general_system_template = """
    # You are an attentive and thorough AI assistant, supporting healthcare professionals by identifying the most relevant clinical studies for their patients.
    # Use the following pieces of context to answer the question at the end.
    # A doctor will enter information about a patient and their condition, and you will need to find the most relevant clinical studies for that patient.
    # The context contains several potential studies for that patient.
    # When you find particular study in the context useful, make sure to cite it in the answer using the link and study identifier.
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # ----
    # {summaries}
    # ----
    # Your mission is not to summarize, but to critically analyze each study based on the patient's condition and requirements.
    # Initiate the dialogue by seeking pertinent details such as histopathological markers, medical history, or other specifics, which are instrumental for your decision-making.
    # Refrain from recommending a study until you have gathered sufficient information to ascertain its suitability. If information is lacking, continue to probe for relevant data.
    # Your goal is to single out one or two studies that best match the patient's needs, achieved through insightful questioning.
    # If no study is appropriate, clearly communicate this.
    # Maintain a strict focus on the patient's information as shared by the professional, and do not consider studies pertaining to unrelated conditions.
    # Under no circumstances show information for studies that do not suit the patient. Respond professionally, matching the language used in the doctor's information.
    # Generate concise answers with references sources section of links to
    # relevant clinical trial information only at the end of the answer. Always answer in the language you were queried.
    # """
    general_system_template = """You are a helpful AI agent that names all potential studies for a patient.
    In the following you get a summary of all studies that suite your patient. Name them and try to match them to the patient. Ask for mor information as needed.
    If you don't know the answer, just say that you don't know, you must not make up an answer. Answer in the language you were queried.
    Study summaries that might be suitable:
    ----
    {summaries}
    ----
    """

    general_user_template = (
        "Patient description from the professional:```{question}```"
    )
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    # Vector + Knowledge Graph response
    kg = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="neo4j",  # neo4j by default
        index_name="study_data",  # vector by default
        text_node_property="body",  # text by default
        retrieval_query="""
MATCH (s:Study)
WITH s, s.score AS similarity
CALL {
    WITH s
    OPTIONAL MATCH (s)-[:HAS_INDICATION]->(ind:Indication)
    OPTIONAL MATCH (s)-[:HAS_SUBINDICATION]->(sub:SubIndication)
    OPTIONAL MATCH (s)-[relContact:HAS_CONTACT]->(con:Contact)
    OPTIONAL MATCH (s)-[:CONDUCTED_AT]->(sc:StudyCenter)
    OPTIONAL MATCH (s)-[:HAS_CRITERIA]->(criteria:Criteria)
    WITH s, ind, sub, 
        COLLECT(DISTINCT con.name) AS ContactNames, 
        COLLECT(DISTINCT relContact.email) AS ContactEmails,
        COLLECT(DISTINCT sc.name) AS StudyCenterNames, 
        COLLECT(DISTINCT criteria.description) AS CriteriaValues
    RETURN 
    '##Study Name: ' + s.name + 
    '##Criteria: ' + REDUCE(cr = '', val IN CriteriaValues | cr + val + '; ') +
    '##Conducted at: ' + REDUCE(sc = '', name IN StudyCenterNames | sc + name + '; ')
    AS text
} 
WITH text, s, similarity
RETURN text,
       similarity as score,
       {source: s.metadata} AS metadata
ORDER BY similarity ASC



""",
    )
    #       s.identifier AS StudyIdentifier,
    #    COALESCE(ind.name, "N/A") AS Indication,
    #    COALESCE(sub.name, "N/A") AS SubIndication,
    # s.embedding AS embedding,

    # '\nIdentifier: ' + study.identifier +
    # COALESCE('\nIndication: ' + ind.name, '') +
    # COALESCE('\nSub-Indication: ' + sub.name, '') +
    # '\nContact Names: ' + reduce(str='', name IN ContactNames | str + name + '; ') +
    # '\nContact Emails: ' + reduce(str='', email IN ContactEmails | str + email + '; ') +
    # '\nStudy Center Names: ' + reduce(str='', name IN StudyCenterNames | str + name + '; ') +
    # '\nCriteria Values: ' + reduce(str='', value IN CriteriaValues | str + value + '; ') AS text, ind, sub

    # kg = Neo4jVector.from_existing_graph(
    #     embedding=embeddings,
    #     url=embeddings_store_url,
    #     username=username,
    #     password=password,
    #     node_label="Study",
    #     text_node_properties=["name"],
    #     embedding_node_property="embedding",
    # )

    kg_qa = RetrievalQAWithSourcesChain(  # TODO:Optimize
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(search_kwargs={"k": 3}),
        reduce_k_below_max_tokens=True,
        max_tokens_limit=3375,
        verbose=True,
        memory=memory,
    )
    return kg_qa
