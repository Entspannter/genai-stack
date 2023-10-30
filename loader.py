import os
from dotenv import load_dotenv
from langchain.graphs import Neo4jGraph
import streamlit as st
from streamlit.logger import get_logger
from chains import load_embedding_model
from utils import create_constraints, create_vector_index
import traceback
import pandas as pd
import uuid

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


def add_embeddings_to_study(study_dict: dict) -> dict:
    """
    Add embeddings to the study dictionary by considering all its fields.
    """
    # Concatenate all the values in the study_dict to form a single string
    study_text = " \n ".join(
        [f"{key}: {str(val)}" for key, val in study_dict.items()]
    )

    # Add embeddings
    study_dict["embedding"] = embeddings.embed_query(study_text)
    study_dict["description"] = study_text

    return study_dict


def generate_unique_id():
    return "STUDYMATCH-" + str(uuid.uuid4())


def load_csv_data(filename: str) -> None:
    # Load CSV into a pandas DataFrame
    df = pd.read_csv(filename)
    df = df.fillna("UNKNOWN_VALUE")

    all_studies = []  # A list to store all study dictionaries with embeddings

    # Iterate through each row (study)
    for _, row in df.iterrows():
        # Split contact and email by ';' if they are strings
        contacts = (
            row["contact"].split(";")
            if isinstance(row["contact"], str)
            else [row["contact"]]
        )
        contact_emails = (
            row["contact_email"].split(";")
            if isinstance(row["contact_email"], str)
            else [row["contact_email"]]
        )

        # Ensure contacts and contact_emails have the same length
        while len(contact_emails) < len(contacts):
            contact_emails.append("UNKNOWN_VALUE")

        study_dict = {
            "name": row["name"],
            "short_name": row["short_name"],
            "identifier": row["identifier"]
            if row["identifier"] != "UNKNOWN_VALUE"
            else generate_unique_id(),
            "study_centers": row["study_centers"].split(";")
            if isinstance(row["study_centers"], str)
            else [row["study_centers"]],
            "indication": row["indication"],
            "sub_indication": row["sub_indication"],
            "criteria": list(
                filter(
                    None,
                    [
                        criterion.strip()
                        for criterion in row["criteria"].split("\n")
                    ],
                )
            ),
            "contacts": contacts,
            "contact_emails": contact_emails,
            "metadata": [
                "stored_by_admin",
            ],  # Add the metadata field here
        }

        # Add embeddings
        study_with_embeddings = add_embeddings_to_study(study_dict)
        all_studies.append(study_with_embeddings)

    # Insert the studies into Neo4j
    insert_csv_data(all_studies)


def insert_csv_data(data: list) -> None:
    import_query = """
    UNWIND $data AS study
    MERGE (s:Study {name: study.name, short_name: study.short_name, description: study.description, embedding: study.embedding, metadata: study.metadata})
    ON CREATE SET s.identifier = study.identifier

    WITH study, s
    UNWIND study.study_centers AS center_name
    MERGE (center:StudyCenter {name: center_name})
    MERGE (s)-[:CONDUCTED_AT]->(center)

    WITH study, s
    UNWIND study.indication AS indication_name
    MERGE (ind:Indication {name: indication_name})
    MERGE (s)-[:HAS_INDICATION]->(ind)

    WITH study, s
    UNWIND study.sub_indication AS sub_ind_name
    MERGE (subInd:SubIndication {name: sub_ind_name})
    MERGE (s)-[:HAS_SUBINDICATION]->(subInd)

    WITH study, s, study.contacts as contacts, study.contact_emails as emails
    FOREACH (i in range(0, size(contacts) - 1) |
        MERGE (contact:Contact {name: contacts[i]})
        MERGE (s)-[rel:HAS_CONTACT {email: emails[i]}]->(contact)
    )

    WITH study, s
    UNWIND study.criteria AS criteria_item
    MERGE (c:Criteria {description: criteria_item})
    MERGE (s)-[:HAS_CRITERIA]->(c)
    """
    neo4j_graph.query(import_query, {"data": data})


def render_page():
    st.header("Study Loader")
    st.subheader("Upload study data to load into Neo4j")
    st.caption("Go to http://localhost:7474/ to explore the graph.")

    # Check if 'button_pressed' is not in the session state
    if "button_pressed" not in st.session_state:
        st.session_state.button_pressed = False

    if st.button("Import", type="primary"):
        st.session_state.button_pressed = True

    if st.session_state.button_pressed:
        print("Button pressed, attempting to load data...")
        with st.spinner(
            "Importing data... Please wait 🕐"
        ):  # This will show a spinner until the code inside completes
            try:
                print("Inside try block")
                load_csv_data(filename="example_studies/all_studies.csv")
                st.success("Import successful ✅")
                st.caption("Data model")
                st.caption(
                    "Go to http://localhost:7474/ to interact with the database"
                )
                st.session_state.button_pressed = False  # Reset the button state after the action is completed
            except Exception as e:
                print("test print")
                st.error(f"Error: {e} 🚨")
                st.error(traceback.format_exc())
                st.error(f"WDIR: {os.getcwd()}")
                st.error(f"DIR LS: {os.listdir()}")


render_page()
