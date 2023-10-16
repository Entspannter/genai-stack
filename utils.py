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
    raise Exception(
        f"Failed to run query after {retries} attempts due to deadlock."
    )


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
