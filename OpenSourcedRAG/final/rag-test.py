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
    device_map="auto",
    max_new_tokens=400  # Increase the maximum number of tokens generated
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
        '''You are a helpful assistant that gives direct, concise answers.
        Use the following context to answer the question, but do not repeat or reference the context.
        Keep your answer short and to the point.
        
        Context: {context}
        ''',
    ),
    ("human", "{question}"),
])

# Create the QA chain using the retriever to get the context dynamically
def qa_chain(query):
    embedding = embedding_model._client.encode(query, prompt_name="s2p_query")
    embedding = embedding.tolist()
    res = retriever.similarity_search_with_score_by_vector(embedding, k=3)
    
    # Use the top 2 documents for context
    context = "\n\n".join([doc.page_content for doc, score in res[:2]])
    
    # Define the prompt template
    prompt_template = """
        You are a helpful assistant that gives direct, concise answers.
        Use the following context to answer the question, but do not repeat or reference the context.
        Keep your answer short and to the point.
        
        Context:
        {context}

        Question:
        {question}

        Answer:
        """.strip()
    
    # Format the prompt with context and question
    prompt_text = prompt_template.format(context=context, question=query)
    
    # Generate the answer
    result = llm(prompt_text)
    
    # Check if the result contains the expected answer
    if "Answer:" in result:
        # Extract the answer after the "Answer:" keyword
        answer_start = result.index("Answer:") + len("Answer:")
        answer = result[answer_start:].strip().split('\n')[0]
    else:
        answer = "No answer found."
    
    # Print the similarity scores and document names for the top 2 documents
    print("\nRetrieved documents (similarity scores):")
    for i, (doc, score) in enumerate(res[:2], 1):
        doc_name = doc.metadata.get('name', 'Unknown document')
        print(f"Doc {i}: {doc_name} (score: {str(score)})")
    
    return answer

while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    result = qa_chain(query)  # Removed top_docs
    print("\nAnswer:", result)  # Only print the answer
