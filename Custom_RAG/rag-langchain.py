import redis
import numpy as np
from time import time
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
from torch import bfloat16
import torch
import transformers

from langchain.vectorstores import Redis
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from huggingface_hub import login


# HuggingFace model and tokenizer setup
model_id = "meta-llama/Llama-3.1-8B-Instruct"
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
token = "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"

login(token=token)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# Load model
model_config = AutoConfig.from_pretrained(
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

# Create HuggingFace pipeline
query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        max_length=1024,
        device_map="auto"
)

# redis connection
redis_host = "redis-18847.c16.us-east-1-2.ec2.redns.redis-cloud.com"
redis_port = 18847  # Replace with your port
redis_password = "ZhIcpw65p96DNC1wtxiftOuGwLI73Qpm"  # Replace with your password

redis_conn = redis.StrictRedis(
    host=redis_host,
    port=redis_port,
    password=redis_password,
    decode_responses=True
)
redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}"

# Wrap the model pipeline for LangChain
llm = HuggingFacePipeline(pipeline=query_pipeline)

# Initialize LangChain Redis retriever for querying Redis-based vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Redis(redis_url=redis_url, index_name="umiacs", embedding=embeddings)

# Define a prompt template (can adjust for more complex query)
template = """
You are given a question and some relevant context. Use the context to generate a response.

Context: {context}
Question: {question}
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Set up the retrieval-based QA system
qa_chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever(), prompt=prompt)

# Test the system
message = "How to use the printer?"
result = qa_chain.run(message)
print(result)
