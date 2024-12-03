from langchain_qdrant import QdrantVectorStore
from OurParentDocumentRetriever import OurParentDocumentRetriever
from langchain_huggingface import HuggingFaceEmbeddings
import pickle

embedding_model = HuggingFaceEmbeddings(
    model_name="dunzhang/stella_en_1.5B_v5",
    model_kwargs={
        "trust_remote_code": True,
    },
)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model, collection_name="parentdoc", path="./qdrant"
)

store = None
with open("parentdoc.pkl", "rb") as f:  # you need the file to be there
    store = pickle.load(f)

retriever = OurParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=store,
)


query = ""
while True:
    print("\n\n\n")

    query = input("bye or query: ")
    # query = "how do I run a python notebook in nexus?"
    if query == "bye":
        break

    res = retriever.similarity_search_with_score(query, k=4)
    print("no prompt embedding:", [(x[0].metadata["name"], x[1]) for x in res])
    # ok... https://github.com/langchain-ai/langchain/commit/0640cbf2f126f773b7ae78b0f94c1ba0caabb2c1
    embedding = embedding_model._client.encode(query, prompt_name="s2p_query")
    embedding = embedding.tolist()
    res = retriever.similarity_search_with_score_by_vector(embedding, k=4)
    print("s2p embedding:", [(x[0].metadata["name"], x[1]) for x in res])
    context = [x[0].page_content for x in res]
    # print(f"Context (using s2p prompt):\n", *context, sep="\n")
