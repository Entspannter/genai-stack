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
        result = tx.run("CALL db.indexes() YIELD name WHERE name = 'index_session_id' RETURN count(name) as count")
        count = result.single()["count"]
        
        # If the index doesn't exist, create it
        if count == 0:
            tx.run("CREATE INDEX index_session_id FOR (m:Message) ON (m.session_id)")

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
        query = (
            "MATCH (m:Message) WHERE m.session_id = $session_id RETURN m.message ORDER BY m.timestamp"
        )
        return tx.run(query, session_id=session_id).data()

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in Neo4j"""
        with self._driver.session() as session:
            session.write_transaction(self._add_message_tx, _message_to_dict(message), self.session_id)

    @staticmethod
    def _add_message_tx(tx, message_data, session_id):
        query = (
            "CREATE (m:Message { session_id: $session_id, message: $message_data, timestamp: datetime() })"
        )
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