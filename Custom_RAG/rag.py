import redis
import sys
from torch import bfloat16
import torch
import transformers
from transformers import AutoTokenizer
from time import time
#import chromadb
#from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

client = redis.StrictRedis(host='localhost', port=6379, db=0)

if client.ping():
    print("Connected to Redis")

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'


# pipeline = transformers.pipeline(
#   "text-generation",
#   model=model_id,
#   model_kwargs={"torch_dtype": torch.float16},
#   device="cpu",
# )

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
    max_new_tokens=1024
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

def test_model(tokenizer, pipeline, message):
    """
    Perform a query
    print the result
    Args:
        tokenizer: the tokenizer
        pipeline: the pipeline
        message: the prompt
    Returns
        None
    """    
    time_start = time()
    sequences = pipeline(
        message,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,)
    time_end = time()
    total_time = f"{round(time_end-time_start, 3)} sec."
    
    question = sequences[0]['generated_text'][:len(message)]
    answer = sequences[0]['generated_text'][len(message):]
    
    return f"Question: {question}\nAnswer: {answer}\nTotal time: {total_time}"


# response = test_model(tokenizer,
#                     query_pipeline,
#                    "Question")
