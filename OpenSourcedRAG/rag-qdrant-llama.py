# Code was made with the help of OpenAI LLM, I do not claim I wrote this code.
import os
import qdrant_client
from transformers import AutoTokenizer, pipeline
import torch
import transformers
from config import get_llama_token, get_qdrant_api_key, get_qdrant_endpoint_url, get_local_folder_path, get_collection_name
import numpy as np

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
    # Dummy function to convert a document to a vector
    # Replace with your actual embedding logic
    return [0.0] * 768  # Example vector of size 768

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

query_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    max_length=1024,
    device_map="auto",
)

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

message = input("Enter your message: ")
# Tokenize in smaller batches
def tokenize_in_batches(texts, batch_size=8):
    for i in range(0, len(texts), batch_size):
        yield tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True).to(device)

# Example usage of batch tokenization
message_batches = list(tokenize_in_batches([message]))

# Generate the query embedding in batches
query_embeddings = []
for batch in message_batches:
    outputs = query_pipeline.model(**batch, output_hidden_states=True)
    query_embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().detach().numpy()
    query_embeddings.append(query_embedding)

# Combine embeddings if necessary
query_embedding = np.mean(query_embeddings, axis=0)

# Retrieve most similar documents from Qdrant
print("Searching for most similar documents...")
relevant_docs = find_most_similar_docs(client, query_embedding, top_k=5)
print(f"Found {len(relevant_docs)} similar documents.")

# Concatenate the retrieved documents with the message
print("Concatenating retrieved documents with the message...")
context = " ".join([client.get_document(get_collection_name(), doc_id).payload["content"] for doc_id in relevant_docs])
full_message = context + "\n" + message
print("Full message prepared.")

# Generate the response using the full context and message
print("Generating response...")
sequences = query_pipeline(
    full_message,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)

question = sequences[0]['generated_text'][:len(message)]
answer = sequences[0]['generated_text'][len(message):]

print("Question:", question)
print("Answer:", answer)
