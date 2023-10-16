import os
from dotenv import load_dotenv
from langchain.graphs import Neo4jGraph
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from streamlit.logger import get_logger
from chains import load_embedding_model
from utils import create_constraints, create_vector_index
import traceback

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

embeddings, dimension = load_embedding_model(
    embedding_model_name,
    config={"ollama_base_url": ollama_base_url},
    logger=logger,
)

# if Neo4j is local, you can go to http://localhost:7474/ to browse the database
neo4j_graph = Neo4jGraph(url=url, username=username, password=password)

create_constraints(neo4j_graph)
create_vector_index(neo4j_graph, dimension)


def load_csv_data(filename=str) -> None:
    csv_loader = CSVLoader(filename)
    data = csv_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    # Extract raw text from Document objects
    raw_texts = [document.page_content for document in data]
    st.text(raw_texts)

    # Split the texts
    split_documents = text_splitter.create_documents(raw_texts)

    # Extract raw text from split Document objects
    split_texts = [document.page_content for document in split_documents]

    # Check type of split_texts for debugging
    st.text(
        type(split_texts[0])
    )  # This should ideally print <class 'str'> for all elements

    # Embed the split texts
    embedding_results = embeddings.embed_documents(split_texts)

    # Add embeddings back to original data. You might have to adjust this based on the actual structure and requirement of your 'data'
    for item, embedding in zip(data, embedding_results):
        item.embedding = embedding

    insert_csv_data(data)


def insert_csv_data(data: list) -> None:
    import_query = """
    UNWIND $data AS study
    MERGE (s:Study {name: study.name, short_name: study.short_name, embedding: study.embedding})
    ON CREATE SET s.identifier = study.identifier
    MERGE (center:StudyCenter {name: study.study_centers})
    MERGE (ind:Indication {name: study.indication})
    MERGE (subInd:SubIndication {name: study.sub_indication})
    MERGE (contact:Contact {name: study.contact, email: study.contact_email})

    MERGE (s)-[:CONDUCTED_AT]->(center)
    MERGE (s)-[:HAS_INDICATION]->(ind)
    MERGE (s)-[:HAS_SUBINDICATION]->(subInd)
    MERGE (s)-[:HAS_CONTACT]->(contact)

    WITH study.criteria AS criteria_list, s
    UNWIND split(criteria_list, "\n") AS criteria_item
    MERGE (c:Criteria {description: criteria_item.trim()})
    MERGE (s)-[:HAS_CRITERIA]->(c)
    """
    print("Made it here")
    neo4j_graph.query(import_query, {"data": data})
    print("Query succesful")


def render_page():
    st.header("Study Loader 4")
    st.subheader("Upload study data to load into Neo4j")
    st.caption("Go to http://localhost:7474/ to explore the graph.")
    # Check if 'button_pressed' is not in the session state
    if "button_pressed" not in st.session_state:
        st.session_state.button_pressed = False

    if st.button("Import", type="primary"):
        st.session_state.button_pressed = True

    if st.session_state.button_pressed:
        print("Button pressed, attempting to load data...")
        try:
            print("Inside try block")
            load_csv_data(filename="example_studies/all_studies.csv")
            st.success("Import successful ✅")
            st.caption("Data model")
            st.caption(
                "Go to http://localhost:7474/ to interact with the database"
            )
            st.session_state.button_pressed = (
                False  # Reset the button state after the action is completed
            )
        except Exception as e:
            print("test print")
            st.error(f"Error: {e} 🚨")
            st.error(traceback.format_exc())
            st.error(f"WDIR: {os.getcwd()}")
            st.error(f"DIR LS: {os.listdir()}")


render_page()
