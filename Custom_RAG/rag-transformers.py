import redis
import sys
from torch import bfloat16
import torch
import transformers
from transformers import AutoTokenizer
from time import time
# from retrieve import find_most_similar_docs

#if connected to redis locally
#client = redis.StrictRedis(host='localhost', port=6379, db=0)

client = redis.Redis(
  host='*',
  port=16332,
  password='*')

model_id = "meta-llama/Llama-3.1-8B-Instruct"
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
token = "*"

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

print(device)

time_start = time()
model_config = transformers.AutoConfig.from_pretrained(
   model_id,
    trust_remote_code=True,
    max_new_tokens=1024,
    use_auth_token=token
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
time_end = time()
print(f"Prepare model, tokenizer: {round(time_end-time_start, 3)} sec.")

#query pipeline
time_start = time()
query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        max_length=1024,
        device_map="auto",)
time_end = time()
print(f"Prepare pipeline: {round(time_end-time_start, 3)} sec.")

import numpy as np

def get_embedding_from_redis(client, doc_id):
    """
    Retrieve document embedding from Redis.
    Args:
        client: Redis client
        doc_id: Document identifier
    Returns:
        numpy array of the embedding
    """
    embedding_bytes = client.hget(f"doc:{doc_id}", "embedding")
    return np.frombuffer(embedding_bytes, dtype=np.float32)

def find_most_similar_docs(client, query_embedding, top_k=5):
    """
    Find the most similar documents by comparing the query embedding with stored embeddings.
    Args:
        client: Redis client
        query_embedding: The query embedding
        top_k: Number of top similar documents to retrieve
    Returns:
        List of document IDs corresponding to the most similar documents
    """
    doc_ids = [key.decode("utf-8") for key in client.keys("doc:*")]
    similarities = []
    
    for doc_id in doc_ids:
        doc_embedding = get_embedding_from_redis(client, doc_id)
        # Compute cosine similarity
        similarity = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
        similarities.append((doc_id, similarity))
    
    # Sort by similarity and return top_k document IDs
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in similarities[:top_k]]

def test_model_with_redis(client, tokenizer, pipeline, message, top_k=5):
    """
    Perform a query, retrieve relevant documents from Redis, and print the result.
    Args:
        client: Redis client
        tokenizer: the tokenizer
        pipeline: the pipeline
        message: the prompt
    Returns
        None
    """
    # Generate the query embedding
    inputs = tokenizer(message, return_tensors="pt")
    query_embedding = pipeline.model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().detach().numpy()
    
    # Retrieve most similar documents from Redis
    relevant_docs = find_most_similar_docs(client, query_embedding, top_k=top_k)
    
    # Concatenate the retrieved documents with the message
    context = " ".join([client.hget(doc_id, "content").decode("utf-8") for doc_id in relevant_docs])
    full_message = context + "\n" + message
    
    # Generate the response using the full context and message
    time_start = time()
    sequences = pipeline(
        full_message,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    time_end = time()
    total_time = f"{round(time_end - time_start, 3)} sec."
    
    question = sequences[0]['generated_text'][:len(message)]
    answer = sequences[0]['generated_text'][len(message):]
    
    return f"Question: {question}\nAnswer: {answer}\nTotal time: {total_time}"

message="What is UMIACS?"
test_model_with_redis(client=client, tokenizer=tokenizer, pipeline=query_pipeline, message=message)

