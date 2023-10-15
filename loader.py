import os
import requests
from dotenv import load_dotenv
from langchain.graphs import Neo4jGraph
from langchain.document_loaders import CSVLoader
import streamlit as st
from streamlit.logger import get_logger
from chains import load_embedding_model
from utils import create_constraints, create_vector_index
from PIL import Image

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

so_api_base_url = "https://api.stackexchange.com/2.3/search/advanced"

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
    print("WTF")
    csv_loader = CSVLoader(filename)
    print("initiating data loading")
    data = csv_loader.load()
    print(data)
    insert_csv_data(data)


def insert_csv_data(data: list) -> None:
    import_query = """
    UNWIND $data AS study
    MERGE (s:Study {name: study.name, short_name: study.short_name})
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
    st.header("Study Loader")
    st.subheader("Upload study data to load into Neo4j")
    st.caption("Go to http://localhost:7474/ to explore the graph.")
    if st.button("Import", type="primary"):
        with st.spinner("Loading... This might take a minute or two."):
            try:
                load_csv_data(filename="example_studies/all_studies.csv")
                st.success("Import successful", icon="✅")
                st.caption("Data model")
                st.caption(
                    "Go to http://localhost:7474/ to interact with the database"
                )
            except Exception as e:
                st.error(f"Error: {e}", icon="🚨")


render_page()
