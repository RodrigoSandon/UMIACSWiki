# Imports (same as before)
import redis
import numpy as np
from time import time
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
from torch import bfloat16
import torch
import transformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings
import pickle as pkl
from langchain_qdrant import QdrantVectorStore
from OurParentDocumentRetriever import OurParentDocumentRetriever


# HuggingFace model and tokenizer setup
model_id = "meta-llama/Llama-3.1-8B-Instruct"
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
token = "*"

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

# Define a prompt template
template = """
You are given a question and some relevant context. Use the context to generate a response.

Context: {context}
Question: {question}
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Set up the retrieval-based QA system using your custom retriever
qa_chain = RetrievalQA.from_llm(llm=llm, retriever=retriever, prompt=prompt)

# Test the system
message = "How to use the printer?"
result = qa_chain.run(message)
print(result)
