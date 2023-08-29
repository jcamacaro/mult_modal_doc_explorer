"""VectorStore wrapper around a Postgres/PGVector database."""
from __future__ import annotations

import enum
import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import numpy as np

import sqlalchemy
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Session, declarative_base, relationship

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore

Base = declarative_base()  # type: Any


_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"


class BaseModel(Base):
    __abstract__ = True
    uuid = sqlalchemy.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class CollectionStore(BaseModel):
    __tablename__ = "langchain_pg_collection"

    name = sqlalchemy.Column(sqlalchemy.String)
    cmetadata = sqlalchemy.Column(JSON)

    embeddings = relationship(
        "EmbeddingStore",
        back_populates="collection",
        passive_deletes=True,
    )

    @classmethod
    def get_by_name(cls, session: Session, name: str) -> Optional["CollectionStore"]:
        return session.query(cls).filter(cls.name == name).first()  # type: ignore

    @classmethod
    def get_or_create(
        cls,
        session: Session,
        name: str,
        cmetadata: Optional[dict] = None,
    ) -> Tuple["CollectionStore", bool]:
        """
        Get or create a collection.
        Returns [Collection, bool] where the bool is True if the collection was created.
        """
        created = False
        collection = cls.get_by_name(session, name)
        if collection:
            return collection, created

        collection = cls(name=name, cmetadata=cmetadata)
        session.add(collection)
        session.commit()
        created = True
        return collection, created


class EmbeddingStore(BaseModel):
    __tablename__ = "images_pg_embedding"

    collection_id = sqlalchemy.Column(
        UUID(as_uuid=True),
        sqlalchemy.ForeignKey(
            f"{CollectionStore.__tablename__}.uuid",
            ondelete="CASCADE",
        ),
    )
    collection = relationship(CollectionStore, back_populates="embeddings")

    embedding: Vector = sqlalchemy.Column(Vector(None))
    document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    cmetadata = sqlalchemy.Column(JSON, nullable=True)

    # custom_id : any user defined id
    custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)


class QueryResult:
    EmbeddingStore: EmbeddingStore
    distance: float


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = EmbeddingStore.embedding.l2_distance
    COSINE = EmbeddingStore.embedding.cosine_distance
    MAX_INNER_PRODUCT = EmbeddingStore.embedding.max_inner_product


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.EUCLIDEAN


class IPGVector(VectorStore):
    """
    VectorStore implementation using Postgres and pgvector.
    - `connection_string` is a postgres connection string.
    - `embedding_function` any embedding function implementing
        `langchain.embeddings.base.Embeddings` interface.
    - `collection_name` is the name of the collection to use. (default: langchain)
        - NOTE: This is not the name of the table, but the name of the collection.
            The tables will be created when initializing the store (if not exists)
            So, make sure the user has the right permissions to create tables.
    - `distance_strategy` is the distance strategy to use. (default: EUCLIDEAN)
        - `EUCLIDEAN` is the euclidean distance.
        - `COSINE` is the cosine distance.
    - `pre_delete_collection` if True, will delete the collection if it exists.
        (default: False)
        - Useful for testing.
    """

    def __init__(
        self,
        connection_string: str,
        embedding_function : Optional = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[dict] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        logger.info("--------->---------->")
        self.connection_string = connection_string
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata
        self.distance_strategy = distance_strategy
        self.pre_delete_collection = pre_delete_collection
        self.logger = logger
        # self.__post_init__()


    def __post_init__(
        self,
    ) -> None:
        """
        Initialize the store.
        """
        self._conn = self.connect()
        # self.create_vector_extension()
        self.create_tables_if_not_exists()
        self.create_collection()

    def connect(self) -> sqlalchemy.engine.Connection:
        engine = sqlalchemy.create_engine(self.connection_string)
        conn = engine.connect()
        return conn

    def create_vector_extension(self) -> None:
        try:
            with Session(self._conn) as session:
                statement = sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector")
                session.execute(statement)
                session.commit()
        except Exception as e:
            self.logger.exception(e)

    def create_tables_if_not_exists(self) -> None:
        with self._conn.begin():
            Base.metadata.create_all(self._conn)

    def drop_tables(self) -> None:
        with self._conn.begin():
            Base.metadata.drop_all(self._conn)

    def create_collection(self) -> None:
        if self.pre_delete_collection:
            self.delete_collection()
        with Session(self._conn) as session:
            CollectionStore.get_or_create(
                session, self.collection_name, cmetadata=self.collection_metadata
            )

    def delete_collection(self) -> None:
        self.logger.debug("Trying to delete collection")
        with Session(self._conn) as session:
            collection = self.get_collection(session)
            if not collection:
                self.logger.warning("Collection not found")
                return
            session.delete(collection)
            session.commit()

    def get_collection(self, session: Session) -> Optional["CollectionStore"]:
        return CollectionStore.get_by_name(session, self.collection_name)

    @classmethod
    def __from(
        cls,
        images,
        embeddings,
        embedding = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        pre_delete_collection: bool = False,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ) -> IPGVector:
        logger.info("Paso por __from")
        connection_string = cls.get_connection_string(kwargs)
        logger.info(f"-----> the connection string {connection_string} ")
        logger.info(f"-----> the collection name {collection_name} ")
        logger.info(f"-----> the embedding {embedding} ")
        logger.info(f"-----> the distance strategy {distance_strategy} ")
        logger.info(f"-----> predelete collection {pre_delete_collection} ")

        store = cls(connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            logger=logger)

        store.test_working(logger)

        # store.logger = logger
        # store.connection_string = connection_string
        # store = cls(
        #     connection_string=connection_string,
        #     collection_name=collection_name,
        #     embedding_function=embedding,
        #     distance_strategy=distance_strategy,
        #     pre_delete_collection=pre_delete_collection,
        #     logger=logger
        # )

        ids = store.add_embeddings(
            images=images, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        # logger.info(ids)



        return None

    def add_embeddings(
        self,
        images,
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """

        self.logger.info("#############################################")
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in images]

        if not metadatas:
            metadatas = [{} for _ in images]

        with Session(self._conn) as session:
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")
            for image, metadata, embedding, id in zip(images, metadatas, embeddings, ids):
                embedding_store = EmbeddingStore(
                    embedding=embedding,
                    document=image,
                    cmetadata=metadata,
                    custom_id=id,
                    collection_id=collection.uuid,
                )
                session.add(embedding_store)
            session.commit()

        return ids

    def add_image(
        self,
        image,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = self.embedding_function.embed_documents(list(image))
        return self.add_embeddings(
            texts=image, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with PGVector with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding_function.embed_query(text=query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding_function.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return docs

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        with Session(self._conn) as session:
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")

            filter_by = EmbeddingStore.collection_id == collection.uuid

            if filter is not None:
                filter_clauses = []
                for key, value in filter.items():
                    IN = "in"
                    if isinstance(value, dict) and IN in map(str.lower, value):
                        value_case_insensitive = {
                            k.lower(): v for k, v in value.items()
                        }
                        filter_by_metadata = EmbeddingStore.cmetadata[key].astext.in_(
                            value_case_insensitive[IN]
                        )
                        filter_clauses.append(filter_by_metadata)
                    else:
                        filter_by_metadata = EmbeddingStore.cmetadata[
                            key
                        ].astext == str(value)
                        filter_clauses.append(filter_by_metadata)

                filter_by = sqlalchemy.and_(filter_by, *filter_clauses)

            results: List[QueryResult] = (
                session.query(
                    EmbeddingStore,
                    self.distance_strategy(embedding).label("distance"),  # type: ignore
                )
                .filter(filter_by)
                .order_by(sqlalchemy.asc("distance"))
                .join(
                    CollectionStore,
                    EmbeddingStore.collection_id == CollectionStore.uuid,
                )
                .limit(k)
                .all()
            )

        docs = [
            (
                Document(
                    page_content=result.EmbeddingStore.document,
                    metadata=result.EmbeddingStore.cmetadata,
                ),
                result.distance if self.embedding_function is not None else None,
            )
            for result in results
        ]
        return docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]


    @classmethod
    def from_embeddings(
        cls,
        images_embeddings,
        embedding,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ) -> IPGVector:
        """Construct PGVector wrapper from raw documents and pre-
        generated embeddings.

        Return VectorStore initialized from documents and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the PGVECTOR_CONNECTION_STRING environment variable.

        Example:
            .. code-block:: python

                from langchain import PGVector
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                faiss = PGVector.from_embeddings(text_embedding_pairs, embeddings)
        """
        logger.info("++++++++++++++++++++++++++++")
        texts = [images_embeddings[0]]
        embeddings = [images_embeddings[1]]
        cls.logger = logger



        return None

        # return cls.__from(
        #     texts,
        #     embeddings,
        #     embedding,
        #     metadatas=metadatas,
        #     ids=ids,
        #     collection_name=collection_name,
        #     distance_strategy=distance_strategy,
        #     pre_delete_collection=pre_delete_collection,
        #     logger = logger,
        #     **kwargs,
        # )

    @classmethod
    def from_existing_index(
        cls: Type[IPGVector],
        embedding: Embeddings,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> IPGVector:
        """
        Get intsance of an existing PGVector store.This method will
        return the instance of the store without inserting any new
        embeddings
        """

        connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
        )

        return store

    @classmethod
    def get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
        connection_string: str = get_from_dict_or_env(
            data=kwargs,
            key="connection_string",
            env_key="PGVECTOR_CONNECTION_STRING",
        )

        if not connection_string:
            raise ValueError(
                "Postgres connection string is required"
                "Either pass it as a parameter"
                "or set the PGVECTOR_CONNECTION_STRING environment variable."
            )

        return connection_string

    @classmethod
    def test_if_working(cls):
        cls.logger.info("test ------------------------>")

    def test_working(self, logr):
        self.logger = logr
        self.logger.info("test <----------------------")

    @classmethod
    def connection_string_from_db_params(
        cls,
        driver: str,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> str:
        """Return connection string from database parameters."""
        return f"postgresql+{driver}://{user}:{password}@{host}:{port}/{database}"
