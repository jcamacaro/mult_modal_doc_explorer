import uuid
import enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import sqlalchemy
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Session, declarative_base, relationship, mapped_column
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
# from sqlalchemy_utils import database_exists, create_database



started = False

Base = declarative_base()  # type: Any


_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"


class BaseModel(Base):
    __abstract__ = True
    uuid = sqlalchemy.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class IEmbeddingStore(BaseModel):
    __tablename__ = "images_pg_vectors"

    embedding: Vector = sqlalchemy.Column(Vector(None))
    image =  sqlalchemy.Column(sqlalchemy.String, nullable=True)
    cmetadata = sqlalchemy.Column(JSON, nullable=True)

    # custom_id : any user defined id
    custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)


class QueryResult:
    EmbeddingStore: IEmbeddingStore
    distance: float

class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = IEmbeddingStore.embedding.l2_distance
    COSINE = IEmbeddingStore.embedding.cosine_distance
    MAX_INNER_PRODUCT = IEmbeddingStore.embedding.max_inner_product


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.EUCLIDEAN


def connect(connection_string) -> sqlalchemy.engine.Connection:
    engine = sqlalchemy.create_engine(connection_string)
    conn = engine.connect()
    return conn

def create_vector_extension(connection, logger) -> None:
    try:
        with Session(connection) as session:
            statement = sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector")
            session.execute(statement)
            session.commit()
    except Exception as e:
        logger.exception(e)

def create_tables_if_not_exists(connection) -> None:
    with connection.begin():
        Base.metadata.create_all(connection)



def register_embeddings(connection_str, embeddings, ids=None, metadatas=None, logger=None):
    vectors = embeddings[1]
    images = embeddings[0]


    conn = connect(connection_str)
    create_tables_if_not_exists(conn, logger)
    create_tables_if_not_exists(conn)



    engine = create_engine(connection_str, pool_size=50, echo=False)
    Base.metadata.create_all(engine)
    conn = engine.connect()
    session = sessionmaker(bind=engine)()

    logger.info("------------------------->>>>>>>>")
    logger.info(connection_str)
    logger.info(engine.url)
    logger.info(session)
    logger.info("------------------------->>>>>>>>")

    if ids is None:
        ids = [str(uuid.uuid1()) for _ in vectors]

    if metadatas is None:
        metadatas = [{} for _ in vectors]

    logger.info("------------------------->>>>>>>>")
    logger.info(f"{ids} <<<<<<<<<<<<<<<<<<<<<<<<")
    logger.info(f"{metadatas} <<<<<<<<<<<<<<<<<<<<<<<<")
    logger.info("------------------------->>>>>>>>")

    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"

    collection_name = _LANGCHAIN_DEFAULT_COLLECTION_NAME

    try:
        for fv, image, metadata, id in zip(vectors, images, metadatas, ids):
            embedding_store = IEmbeddingStore(
                embedding=fv,
                image= "...",
                cmetadata=metadata,
                custom_id=id
            )
            session.add(embedding_store)
            logger.info("------------------------->>>>>>>>")
            logger.info(f"Storing <<<<<<<<<<<<<<<<<<<<<<<<")
            logger.info("------------------------->>>>>>>>")
    except:
        logger.info("------------------------->>>>>>>>")
        logger.info(f"Rollback <<<<<<<<<<<<<<<<<<<<<<<<")
        logger.info("------------------------->>>>>>>>")
        session.rollback()
        return False
    else:
        logger.info("------------------------->>>>>>>>")
        logger.info(f"commit <<<<<<<<<<<<<<<<<<<<<<<<")
        logger.info("------------------------->>>>>>>>")
        session.commit()
        return True
