from neo4j.exceptions import TransientError
import time


class BaseLogger:
    def __init__(self) -> None:
        self.info = print


def create_vector_index(driver, dimension: int) -> None:
    index_query = "CALL db.index.vector.createNodeIndex('study_data', 'Study', 'embedding', $dimension, 'cosine')"
    try:
        driver.query(index_query, {"dimension": dimension})
    except:  # Already exists
        pass


MAX_RETRIES = 3


def run_query_with_retries(driver, query, retries=MAX_RETRIES):
    """Run a Neo4j query and handle transient errors with retries."""
    for _ in range(retries):
        try:
            driver.query(query)
            return
        except TransientError as te:
            # Handle deadlock situation by waiting and retrying
            if "DeadlockDetected" in str(te):
                time.sleep(1)  # Wait for a second before retrying
            else:
                raise te
    raise Exception(f"Failed to run query after {retries} attempts due to deadlock.")


def create_constraints(driver):
    # Constraint to ensure unique StudyCenter names
    run_query_with_retries(
        driver,
        "CREATE CONSTRAINT study_center_name IF NOT EXISTS FOR (sc:StudyCenter) REQUIRE sc.name IS UNIQUE",
    )

    # Constraint to ensure unique Indication names
    run_query_with_retries(
        driver,
        "CREATE CONSTRAINT indication_name IF NOT EXISTS FOR (i:Indication) REQUIRE i.name IS UNIQUE",
    )

    # Constraint to ensure unique SubIndication names
    run_query_with_retries(
        driver,
        "CREATE CONSTRAINT subindication_name IF NOT EXISTS FOR (si:SubIndication) REQUIRE si.name IS UNIQUE",
    )

    # Constraint to ensure unique Study identifier, if it exists
    run_query_with_retries(
        driver,
        "CREATE CONSTRAINT study_identifier IF NOT EXISTS FOR (s:Study) REQUIRE s.identifier IS UNIQUE",
    )


from langchain.schema.messages import BaseMessage, _message_to_dict, messages_from_dict
import json
import re
from typing import Any, List
from neo4j import GraphDatabase, basic_auth

from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage, _message_to_dict, messages_from_dict

from neo4j import GraphDatabase
from typing import List


class Neo4jChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Neo4j database."""

    def __init__(self, session_id: str, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.session_id = session_id

    @staticmethod
    def _create_index_if_not_exists(tx):
        # Check if the index already exists
        result = tx.run(
            "CALL db.indexes() YIELD name WHERE name = 'index_session_id' RETURN count(name) as count"
        )
        count = result.single()["count"]

        # If the index doesn't exist, create it
        if count == 0:
            tx.run("CREATE INDEX index_session_id FOR (m:Message) ON (m.session_id)")

    @staticmethod
    def _create_belongs_to_session_relationship(tx, session_id):
        query = (
            "MATCH (m:Message {session_id: $session_id}), (s:Session {id: $session_id}) "
            "MERGE (m)-[:BELONGS_TO_SESSION]->(s)"
        )
        tx.run(query, session_id=session_id)

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from Neo4j"""
        with self._driver.session() as session:
            result = session.read_transaction(self._fetch_messages, self.session_id)
            # Passing the session_id
            items = [record["m.message"] for record in result]
            # Parse each JSON string into a dictionary
            list_of_dicts = [json.loads(s) for s in items]

            messages = messages_from_dict(list_of_dicts)
            return messages

    @staticmethod
    def _fetch_messages(tx, session_id):
        query = "MATCH (m:Message) WHERE m.session_id = $session_id RETURN m.message ORDER BY m.timestamp"
        return tx.run(query, session_id=session_id).data()

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in Neo4j and link it to a session."""
        with self._driver.session() as session:
            session.write_transaction(
                self._add_message_tx, _message_to_dict(message), self.session_id
            )
            # Create relationship to the session
            session.write_transaction(
                self._create_belongs_to_session_relationship, self.session_id
            )

    @staticmethod
    def _add_message_tx(tx, message_data, session_id):
        query = "CREATE (m:Message { session_id: $session_id, message: $message_data, timestamp: datetime() })"
        tx.run(query, session_id=session_id, message_data=json.dumps(message_data))

    def clear(self) -> None:
        """Clear session memory from Neo4j"""
        with self._driver.session() as session:
            session.write_transaction(self._clear_session)

    @staticmethod
    def _clear_session(tx, session_id):
        query = "MATCH (m:Message) WHERE m.session_id = $session_id DELETE m"
        tx.run(query, session_id=session_id)

    def _del_(self) -> None:
        self._driver.close()
