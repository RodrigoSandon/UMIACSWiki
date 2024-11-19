import os
os.environ["HF_HOME"] = "/fs/nexus-projects/umiacs-wiki-chatbot/.cache/huggingfacehub/hub"
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import pickle as pkl
from langchain_qdrant import QdrantVectorStore
from OurParentDocumentRetriever import OurParentDocumentRetriever
import json
from langchain_ollama import ChatOllama


def load():

    llm = ChatOllama(model="llama3.1:8b-instruct-fp16", temperature=.5, num_ctx=32768, keep_alive=-1)
    llm.invoke("") # force ollama to load it asap, shouldn't block the script anyways
    print("llm")

    store = None
    with open("parentdoc.pkl", "rb") as f:
        store = pkl.load(f)
    print("name:content map")

    embedding_model = HuggingFaceEmbeddings(
        model_name="dunzhang/stella_en_1.5B_v5",
        model_kwargs={
            "trust_remote_code": True,
        },
    )
    print("embedding model")

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model, collection_name="parentdoc", path="./qdrant"
    )
    print("vectorstore")

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
                '''
                You are given a question along with relevant background information to guide your answer. 
                Use the context to create a response that is clear, concise, and accurate, incorporating specific details as needed.
                - If the question is informational, respond directly, ensuring alignment with the context.
                - If the question is a request, include a clear response and outline the steps or considerations involved in filing or fulfilling that request.
                - If additional details or assumptions are needed to provide a complete answer, acknowledge these respectfully to maintain clarity and transparency.
                ''',
            ),
            ("human", "Context: {context}\nQuestion: {question}, \n\n Answer the question, remember to only use facts from the context provided. If the question is a request, include a clear response and outline the steps or considerations involved in filing or fulfilling that request."),
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
                    outfile.seek(0)
                    json.dump(outdata, outfile, indent=4)
                    outfile.truncate()
                    answer = f"Error: {str(e)}"
            print(f"{len(outdata.keys())} QA pairs")
            outfile.seek(0)
            json.dump(outdata, outfile, indent=4)
            outfile.truncate()

    input_file = "/fs/nexus-projects/umiacs-wiki-chatbot/tickets-20241031.json"
    output_file = "/fs/nexus-projects/umiacs-wiki-chatbot/dummyname.json"

    # Run the QA pipeline and save the results
    run_qa_pipeline(input_file, output_file)
    print(f"Results saved to {output_file}.txt")
    # print(locals())


if __name__ == "__main__":
    cache = load()
    run(cache)
