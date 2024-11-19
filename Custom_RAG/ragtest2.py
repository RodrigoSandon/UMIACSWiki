from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import pickle as pkl
from langchain_qdrant import QdrantVectorStore
from OurParentDocumentRetriever import OurParentDocumentRetriever
from langchain_groq import ChatGroq
import json


def load():
    os.environ["GROQ_API_KEY"] = (
        "your key"
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    print("llm")

    store = None
    with open("parentdoc.pkl", "rb") as f:
        store = pkl.load(f)

    embedding_model = HuggingFaceEmbeddings(
        model_name="dunzhang/stella_en_1.5B_v5",
        model_kwargs={
            "trust_remote_code": True,
        },
    )

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model, collection_name="parentdoc", path="./qdrant"
    )

    retriever = OurParentDocumentRetriever(vectorstore=vector_store, docstore=store)

    print("retriever")

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
                "You are given a question and some relevant context. Use the context to generate a response.",
            ),
            ("human", "Context: {context}\nQuestion: {question}"),
        ]
    )

    # Create the QA chain using the retriever to get the context dynamically
    def qa_chain(query):
        # Retrieve relevant context from the retriever
        embedding = embedding_model._client.encode(query, prompt_name="s2p_query")
        embedding = embedding.tolist()
        res = retriever.similarity_search_with_score_by_vector(embedding, k=4)
        print("s2p embedding:", [(x[0].metadata["name"], x[1]) for x in res])
        context = [x[0].page_content for x in res]

        chain = prompt | llm

        result = chain.invoke(
            {
                "context": context,
                "question": query,
            }
        )

        return result

    # query = "What is UMIACS?"
    # result = qa_chain(query)
    # print(result)

    def run_qa_pipeline(input_file, output_file):
        with open(input_file, "r") as infile, open(output_file, "w") as outfile:
            indata = json.load(infile)
            outdata = {}
            for line in indata:
                try:
                    answer = qa_chain(line)
                    outdata[line] = answer.content
                except Exception as e:
                    print(e)
                    json.dump(outdata, outfile, indent=4)
                    answer = f"Error: {str(e)}"
                break
            print(outdata)
            # json.dump(outdata, outfile, indent=4)

    input_file = "/fs/nexus-projects/umiacs-wiki-chatbot/tickets-20241031.json"
    output_file = "/fs/nexus-projects/umiacs-wiki-chatbot/ticketresponses1.json"

    # Run the QA pipeline and save the results
    run_qa_pipeline(input_file, output_file)
    print(f"Results saved to {output_file}.txt")
    # print(locals())


if __name__ == "__main__":
    cache = load()
    run(cache)
