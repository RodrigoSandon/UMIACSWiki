from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import CharacterTextSplitter


class OurParentDocumentRetriever(ParentDocumentRetriever):
    # just use the built in one except for retrieval
    # retrieval is just wrapper around langchain vectorstore methods

    def __init__(
        self,
        vectorstore,
        docstore,
        child_splitter=CharacterTextSplitter(),
    ):
        """child_splitter is ONLY optional if using for retrieval"""
        super().__init__(
            vectorstore=vectorstore, docstore=docstore, child_splitter=child_splitter
        )
        # self.docstore = docstore
        # self.vectorstore = vectorstore

        if type(vectorstore).__name__ == "RedisVectorStore":
            self.redisfix = True

        if type(vectorstore).__name__ == "QdrantVectorStore":
            from types import MethodType
            from typing import (
                Any,
                Callable,
                Dict,
                Generator,
                Iterable,
                List,
                Optional,
                Sequence,
                Tuple,
                Type,
                Union,
            )
            from qdrant_client import QdrantClient, models
            from langchain_core.documents import Document

            def implant(
                self,
                embedding: List[float],
                k: int = 4,
                filter: Optional[models.Filter] = None,
                search_params: Optional[models.SearchParams] = None,
                offset: int = 0,
                score_threshold: Optional[float] = None,
                consistency: Optional[models.ReadConsistency] = None,
                **kwargs: Any,
            ) -> List[Document]:
                """Return docs most similar to embedding vector.

                Returns:
                    List of Documents most similar to the query.
                """
                qdrant_filter = filter

                self._validate_collection_for_dense(
                    client=self.client,
                    collection_name=self.collection_name,
                    vector_name=self.vector_name,
                    distance=self.distance,
                    dense_embeddings=embedding,
                )
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=embedding,
                    using=self.vector_name,
                    query_filter=qdrant_filter,
                    search_params=search_params,
                    limit=k,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=score_threshold,
                    consistency=consistency,
                    **kwargs,
                ).points

                return [
                    (
                        self._document_from_point(
                            result,
                            self.collection_name,
                            self.content_payload_key,
                            self.metadata_payload_key,
                        ),
                        result.score,
                    )
                    for result in results
                ]

            self.vectorstore.similarity_search_with_score_by_vector = MethodType(
                implant, self.vectorstore
            )

    def _get_id_doc_map(self, sub_docs):
        # map so we can add scores or something idk
        ids = {}
        for doc in sub_docs:
            # if type(self.vectorstore).__name__ == "RedisVectorStore":  # ?
            if "embedding" in doc.metadata:
                del doc.metadata["embedding"]
            # assumes super().id_key is doc_id ... everything here does actually
            if "doc_id" in doc.metadata and doc.metadata["doc_id"] not in ids:
                id = doc.metadata["doc_id"]
                ids[id] = self.docstore.mget([id])[0]
        return ids

    # there is also maximal_marginal_relevance

    def similarity_search(self, query, k=4, **kwargs):
        """Return documents most similar to query string.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Optional filter expression to apply to the query.
            **kwargs: Other keyword arguments to pass to the search function:
                - custom_query: Optional callable that can be used
                                to customize the query.
                - doc_builder: Optional callable to customize Document creation.
                - return_metadata: Whether to return metadata. Defaults to True.
                - distance_threshold: Optional distance threshold for filtering results.
        """
        sub_docs = None
        if type(self.vectorstore).__name__ == "RedisVectorStore":  # idk
            sub_docs = self.vectorstore.similarity_search(
                query, k, return_all=True, **kwargs
            )
        else:
            sub_docs = self.vectorstore.similarity_search(query, k, **kwargs)

        id_docs = self._get_id_doc_map(sub_docs)
        return list(id_docs.values())

    def similarity_search_by_vector(self, query, k=4, **kwargs):
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Optional filter expression to apply.
            **kwargs: Other keyword arguments:
                - return_metadata: Whether to return metadata. Defaults to True.
                - distance_threshold: Optional distance threshold for filtering results.

        Returns:
            List of Documents most similar to the query vector.
        """
        sub_docs = None
        if type(self.vectorstore).__name__ == "RedisVectorStore":  # idk
            sub_docs = self.vectorstore.similarity_search_by_vector(
                query, k, return_all=True, **kwargs
            )
        else:
            sub_docs = self.vectorstore.similarity_search_by_vector(query, k, **kwargs)
        id_docs = self._get_id_doc_map(sub_docs)
        return list(id_docs.values())

    def similarity_search_with_score(self, query, k=4, **kwargs):
        """Return documents most similar to query string, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Optional filter expression to apply to the query.
            **kwargs: Other keyword arguments to pass to the search function:
                - custom_query: Optional callable that can be used
                                to customize the query.
                - doc_builder: Optional callable to customize Document creation.
                - return_metadata: Whether to return metadata. Defaults to True.
                - distance_threshold: Optional distance threshold for filtering results.
        """
        sub_docs = None
        if type(self.vectorstore).__name__ == "RedisVectorStore":  # idk
            sub_docs = self.vectorstore.similarity_search_with_score(
                query, k, return_all=True, **kwargs
            )
        else:
            sub_docs = self.vectorstore.similarity_search_with_score(query, k, **kwargs)

        id_docs = self._get_id_doc_map([d[0] for d in sub_docs])
        for sub in sub_docs:
            if sub[0].metadata["doc_id"] in id_docs:
                parent = id_docs[sub[0].metadata["doc_id"]]
                if not isinstance(parent, tuple):
                    id_docs[sub[0].metadata["doc_id"]] = (parent, [])
                id_docs[sub[0].metadata["doc_id"]][1].append(sub[1])

        return list(id_docs.values())

    def similarity_search_with_score_by_vector(self, query, k=4, **kwargs):
        """Return documents most similar to embedding vector, along with scores.

        implemented by langchain-redis.redisvectorstore at least

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Optional filter expression to apply to the query.
            **kwargs: Other keyword arguments to pass to the search function:
                - custom_query: Optional callable that can be used
                                to customize the query.
                - doc_builder: Optional callable to customize Document creation.
                - return_metadata: Whether to return metadata. Defaults to True.
                - distance_threshold: Optional distance threshold for filtering results.
        """
        sub_docs = None
        if type(self.vectorstore).__name__ == "RedisVectorStore":  # idk
            sub_docs = self.vectorstore.similarity_search_with_score_by_vector(
                query, k, return_all=True, **kwargs
            )
        else:
            sub_docs = self.vectorstore.similarity_search_with_score_by_vector(
                query, k, **kwargs
            )
        id_docs = self._get_id_doc_map([d[0] for d in sub_docs])
        for sub in sub_docs:
            if sub[0].metadata["doc_id"] in id_docs:
                parent = id_docs[sub[0].metadata["doc_id"]]
                if not isinstance(parent, tuple):
                    id_docs[sub[0].metadata["doc_id"]] = (parent, [])
                id_docs[sub[0].metadata["doc_id"]][1].append(sub[1])

        return list(id_docs.values())
