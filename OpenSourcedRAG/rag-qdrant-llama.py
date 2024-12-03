# Code was made with the help of OpenAI LLM, I do not claim I wrote this code.
import os
import qdrant_client
from transformers import AutoTokenizer, pipeline
import torch
import transformers
from config import get_llama_token, get_qdrant_api_key, get_qdrant_endpoint_url, get_local_folder_path, get_collection_name
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# Init Llama
model_id = "meta-llama/Llama-3.1-8B-Instruct"
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

# Read the Llama token from the env folder
token = get_llama_token()

print(device)

# Define the path to your local folder containing documents
local_folder_path = get_local_folder_path()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Initialize the Qdrant client for the cluster with API key
client = QdrantClient(
    url=get_qdrant_endpoint_url(),
    api_key=get_qdrant_api_key()
)

# Check if the collection already exists
collection_name = get_collection_name()
try:
    client.get_collection(collection_name)
    print(f"Collection '{collection_name}' already exists.")
except Exception as e:
    # If the collection does not exist, create it
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    print(f"Collection '{collection_name}' created.")

from qdrant_client.models import PointStruct

def embed_document(file_path):
    # Load the document content
    with open(file_path, 'r') as file:
        document_content = file.read()
    
    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="dunzhang/stella_en_1.5B_v5",
        model_kwargs={
            "trust_remote_code": True,
        },
    )
    
    # Encode the document content to get the embedding
    embedding = embedding_model._client.encode(document_content, prompt_name="s2p_query")
    
    return embedding.tolist()  # Convert to list if needed

# Iterate over files in the local folder and upload them
for idx, file_name in enumerate(os.listdir(local_folder_path)):
    file_path = os.path.join(local_folder_path, file_name)
    vector = embed_document(file_path)
    payload = {"file_name": file_name}

    # Upsert the document vector into the collection
    client.upsert(
        collection_name=get_collection_name(),
        wait=True,
        points=[
            PointStruct(id=idx, vector=vector, payload=payload)
        ]
    )

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=True,
    max_new_tokens=1024,
    token=token
)

# Function to check available GPU memory
def is_gpu_overloaded(threshold=0.9):
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        return reserved_memory / total_memory > threshold
    return False

# Use mixed precision for the model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    device_map='auto',
    token=token
).half()  # Use half precision

# Check if GPU is overloaded and switch to CPU if necessary
device = 'cuda' if torch.cuda.is_available() and not is_gpu_overloaded() else 'cpu'
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id,token=token)
tokenizer.pad_token = tokenizer.eos_token  # or another appropriate token

query_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    max_length=1024,
    device_map="auto",
)

# Ensure the model is on the GPU
query_pipeline.model.to('cuda')

def find_most_similar_docs(qdrant_client, query_embedding, top_k=5):
    """
    Find the most similar documents by comparing the query embedding with stored embeddings in Qdrant.
    Args:
        qdrant_client: Qdrant client
        query_embedding: The query embedding
        top_k: Number of top similar documents to retrieve
    Returns:
        List of document IDs corresponding to the most similar documents
    """
    search_result = qdrant_client.search(
        collection_name=get_collection_name(),
        query_vector=query_embedding.tolist(),
        limit=top_k
    )
    return [hit.id for hit in search_result]

# Define the wiki prompt to append to each question
wiki_prompt = (
    "You are an AI assistant specifically trained to answer questions based ONLY on the provided wiki page content. "
    "Your knowledge is limited to the information given in the context. Follow these rules strictly:\n\n"
    "- Only use information explicitly stated in the provided context.\n"
    "- If the context doesn't contain relevant information to answer the question, say, 'I don't have enough information to answer that question based on the provided wiki page content.'\n"
    "- Do not use any external knowledge or make assumptions beyond what's in the context.\n"
    "- If asked about topics not covered in the context, state that the wiki page content doesn't cover that topic.\n"
    "- Be precise and concise in your answers, citing specific parts of the context when possible.\n"
    "- If the question is ambiguous or unclear based on the context, ask for clarification.\n"
    "- Never claim to know more than what's provided in the context.\n"
    "- If the context contains conflicting information, point out the inconsistency without resolving it.\n"
    "- Remember, your role is to interpret and relay the information from the wiki page content, not to provide additional knowledge or opinions.\n\n"
)

message = input("Enter your question: ")

# Tokenize the full message
inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True).to('cuda')

# Run the model
outputs = query_pipeline.model(**inputs, output_hidden_states=True)
query_embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().detach().numpy()[:768]

# Retrieve most similar documents from Qdrant
print("Searching for most similar documents...")
relevant_docs = find_most_similar_docs(client, query_embedding, top_k=5)
print(f"Found {len(relevant_docs)} similar documents.")

# Concatenate the retrieved documents with the wiki prompt and question
print("Concatenating retrieved documents with the message...")
context = " ".join([
    (lambda doc_id: (
        print(f"Payload for doc_id {doc_id}: {client.query_points(collection_name=get_collection_name(), query=doc_id, with_payload=True).points[0].payload}") or
        client.query_points(collection_name=get_collection_name(), query=doc_id, with_payload=True).points[0].payload.get("content", "")
    ))(doc_id) for doc_id in relevant_docs
])
full_message_with_context = wiki_prompt + context + "\n" + message  # Include context and question
print("Full message prepared.")

# Generate the response using the full context and message
print("Generating response...")
sequences = query_pipeline(
    full_message_with_context,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=400,
)

question = sequences[0]['generated_text'][:len(message)]
answer = sequences[0]['generated_text'][len(message):]

print("Question:", question)
print("Answer:", answer)
