from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
from torch import bfloat16
import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import pickle as pkl
from langchain_qdrant import QdrantVectorStore
from OurParentDocumentRetriever import OurParentDocumentRetriever
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"]="*"

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
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
    embedding=embedding_model,
    collection_name="parentdoc",
    path="./qdrant" 
)

retriever = OurParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=store
)

print("retriever")

from langchain_core.prompts import ChatPromptTemplate

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
    # context = [x[0].page_content for x in res]
    context = res[0][0].page_content
    
    chain = prompt | llm
    
    result = chain.invoke(
        {
            "context": context,
            "question": query,
        }
    )
    
    return result

query = "I am writing to request additional storage upto 100 GB for the CMSC848F\ncourse I am taking this semester. I have CC'd the TAs for this course and\nmy UMIACS account id is mentioned in the subject line of this email"
result = qa_chain(query)
print(result)

# def run_qa_pipeline(input_file, output_file):
#     with open(input_file, "r") as infile, open(output_file, "w") as outfile:
#         for line in infile:
#             question = line.strip()
            
#             if not question:
#                 continue
            
#             try:
#                 answer = qa_chain(question)
#             except Exception as e:
#                 answer = f"Error: {str(e)}"
            
#             outfile.write(f"Question: {question}\n")
#             outfile.write(f"Answer: {answer}\n")
#             outfile.write("\n")

# input_file = "questions.txt" 
# output_file = "rag-test-results.txt" 

# # Run the QA pipeline and save the results
# run_qa_pipeline(input_file, output_file)
# print("Results saved to rag-test-results.txt")
