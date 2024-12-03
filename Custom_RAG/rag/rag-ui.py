import os
os.environ["HF_HOME"] = "/fs/nexus-projects/umiacs-wiki-chatbot/.cache/huggingfacehub/hub"
import streamlit as st
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import pickle as pkl
from langchain_qdrant import QdrantVectorStore
from OurParentDocumentRetriever import OurParentDocumentRetriever
from langchain_ollama import ChatOllama


def load():
    llm = ChatOllama(model="llama3.1:8b-instruct-fp16", temperature=0.5, num_ctx=32768, keep_alive=-1)
    llm.invoke("")  # Force Ollama to load ASAP
    print("llm loaded")

    store = None
    with open("parentdoc.pkl", "rb") as f:
        store = pkl.load(f)
    print("Document store loaded")

    embedding_model = HuggingFaceEmbeddings(
        model_name="dunzhang/stella_en_1.5B_v5",
        model_kwargs={"trust_remote_code": True},
    )
    print("Embedding model loaded")

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model, collection_name="parentdoc", path="./qdrant"
    )
    print("Vector store initialized")

    retriever = OurParentDocumentRetriever(vectorstore=vector_store, docstore=store)
    print("Retriever initialized")

    return {"embedding_model": embedding_model, "retriever": retriever, "llm": llm}


from langchain_core.prompts import ChatPromptTemplate


def run(cache):
    embedding_model = cache["embedding_model"]
    retriever = cache["retriever"]
    llm = cache["llm"]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''
                You are given a question along with relevant background information to guide your answer. 
                Use the context to create a response that is clear, concise, and accurate, incorporating specific details as needed.
                - If the question is informational, respond directly, ensuring alignment with the context.
                - If the question is a request, include a clear response and outline the steps or considerations involved in filing or fulfilling that request.
                - If additional details or assumptions are needed to provide a complete answer, acknowledge these respectfully to maintain clarity and transparency.
                ''',
            ),
            ("human", "Context: {context}\nQuestion: {question}\n\nAnswer the question, using only facts from the context."),
        ]
    )

    # Create the QA chain using the retriever to get the context dynamically
    def qa_chain(query):
        embedding = embedding_model._client.encode(query, prompt_name="s2p_query").tolist()
        res = retriever.similarity_search_with_score_by_vector(embedding, k=4)
        print("Embeddings and context fetched:", [(x[0].metadata["name"], x[1]) for x in res])
        context = [x[0].page_content for x in res]

        chain = prompt | llm
        result = chain.invoke({"context": context, "question": query})
        return result.content

    # Streamlit UI for the QA process
    st.title("RAG Chatbot")
    st.markdown("Ask any question, and the chatbot will provide responses based on relevant documents.")

    user_input = st.text_area("Enter your question:", placeholder="Type your question here...")
    if st.button("Ask"):
        if user_input.strip():
            with st.spinner("Thinking..."):
                try:
                    answer = qa_chain(user_input)
                    st.success("Chatbot Response:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a question before clicking Ask.")

    st.markdown("---")
    st.caption("Powered by LangChain, Streamlit, and Ollama.")


if __name__ == "__main__":
    cache = load()
    run(cache)
