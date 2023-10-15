class BaseLogger:
    def __init__(self) -> None:
        self.info = print


def create_vector_index(driver, dimension: int) -> None:
    # Vector index for Studies
    index_query_study = "CALL db.index.vector.createNodeIndex('study_data', 'Study', 'embedding', $dimension, 'cosine')"
    try:
        driver.query(index_query_study, {"dimension": dimension})
    except:  # Index already exists
        pass

    # Vector index for Criteria
    index_query_criteria = "CALL db.index.vector.createNodeIndex('criteria_data', 'Criteria', 'embedding', $dimension, 'cosine')"
    try:
        driver.query(index_query_criteria, {"dimension": dimension})
    except:  # Index already exists
        pass

    # Vector index for Indication
    index_query_indication = "CALL db.index.vector.createNodeIndex('indication_data', 'Indication', 'embedding', $dimension, 'cosine')"
    try:
        driver.query(index_query_indication, {"dimension": dimension})
    except:  # Index already exists
        pass


def create_constraints(driver):
    # Constraint to ensure unique StudyCenter names
    driver.query(
        "CREATE CONSTRAINT study_center_name IF NOT EXISTS FOR (sc:StudyCenter) REQUIRE sc.name IS UNIQUE"
    )

    # Constraint to ensure unique Indication names
    driver.query(
        "CREATE CONSTRAINT indication_name IF NOT EXISTS FOR (i:Indication) REQUIRE i.name IS UNIQUE"
    )

    # Constraint to ensure unique SubIndication names
    driver.query(
        "CREATE CONSTRAINT subindication_name IF NOT EXISTS FOR (si:SubIndication) REQUIRE si.name IS UNIQUE"
    )

    # Constraint to ensure unique Study identifier, if it exists
    driver.query(
        "CREATE CONSTRAINT study_identifier IF NOT EXISTS FOR (s:Study) REQUIRE s.identifier IS UNIQUE"
    )
