from transformers import AutoTokenizer, AutoConfig
import os
from langchain_huggingface import HuggingFaceEmbeddings
import pickle as pkl
from langchain_qdrant import QdrantVectorStore
from OurParentDocumentRetriever import OurParentDocumentRetriever
import torch
from langchain_community.llms import HuggingFacePipeline
import transformers
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import warnings
import logging


# Load environment variables from a .env file
load_dotenv()

# HuggingFace model and tokenizer setup
model_id = "meta-llama/Llama-3.1-8B-Instruct"
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
#print("device:", device)
token = os.getenv("llama_instruct_token")

# Suppress HuggingFace warnings
warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load model (add quiet=True)
model_config = AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=True,
    max_new_tokens=2048,
    token=token,
    quiet=True
)

# Use mixed precision for the model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    token=token,
    torch_dtype=torch.float16 # Use float16 instead of calling .half()
)

model.to(device)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=token,
    trust_remote_code=True
)

# Create HuggingFace pipeline
query_pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    max_new_tokens=200,  # Limit the number of tokens to generate
    device_map="auto"
)

# Ensure the model is on the GPU
query_pipeline.model.to('cuda')

# Wrap the model pipeline for LangChain
llm = HuggingFacePipeline(pipeline=query_pipeline)

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

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        '''
        You are a helpful assistant. Provide direct, concise answers based on the given context.
        Do not repeat or reference the context in your response.
        
        Context: {context}
        ''',
    ),
    ("human", "Answer the question: {question}"),
])

# Create the QA chain using the retriever to get the context dynamically
def qa_chain(query):
    embedding = embedding_model._client.encode(query, prompt_name="s2p_query")
    embedding = embedding.tolist()
    res = retriever.similarity_search_with_score_by_vector(embedding, k=3)
    
    # Only use the top document for context
    context = res[0][0].page_content
    chain = prompt | llm
    
    # Now include both context and question in the chain invocation
    result = chain.invoke(
        {
            "context": context,
            "question": query,
        }
    )
    
    # Extract just the answer part after "Answer:"
    answer = str(result)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    
    # Print the similarity scores and document names for each document
    print("\nRetrieved documents (similarity scores):")
    for i, (doc, score) in enumerate(res, 1):
        doc_name = doc.metadata.get('name', 'Unknown document')
        print(f"Doc {i}: {doc_name} (score: {str(score)})")
    
    return answer

while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    result = qa_chain(query)  # Removed top_docs
    print("\nAnswer:", result)  # Only print the answer
