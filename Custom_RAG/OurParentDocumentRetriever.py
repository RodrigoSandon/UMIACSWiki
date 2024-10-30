from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import CharacterTextSplitter


class OurParentDocumentRetriever(ParentDocumentRetriever):
    # just use the built in one except for retrieval
    # retrieval is just wrapper around langchain vectorstore methods

    def __init__(self, vectorstore, docstore, child_splitter=CharacterTextSplitter()):
        """child_splitter is ONLY optional if using for retrieval"""
        super().__init__(
            vectorstore=vectorstore, docstore=docstore, child_splitter=child_splitter
        )
        self.docstore = docstore
        self.vectorstore = vectorstore

    def _get_id_doc_map(self, sub_docs):
        # map so we can add scores or something idk
        ids = {}
        for doc in sub_docs:
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
        sub_docs = self.vectorstore.similarity_search(
            query, k, return_all=True, **kwargs
        )
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
        sub_docs = self.vectorstore.similarity_search_by_vector(
            query, k, return_all=True, **kwargs
        )
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
        sub_docs = self.vectorstore.similarity_search_with_score(
            query, k, return_all=True, **kwargs
        )
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
        sub_docs = self.vectorstore.similarity_search_with_score_by_vector(
            query, k, return_all=True, **kwargs
        )
        id_docs = self._get_id_doc_map([d[0] for d in sub_docs])
        for sub in sub_docs:
            if sub[0].metadata["doc_id"] in id_docs:
                parent = id_docs[sub[0].metadata["doc_id"]]
                if not isinstance(parent, tuple):
                    id_docs[sub[0].metadata["doc_id"]] = (parent, [])
                id_docs[sub[0].metadata["doc_id"]][1].append(sub[1])

        return list(id_docs.values())
