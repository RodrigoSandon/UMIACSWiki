from flask import Flask, request, jsonify
from langchain.llms import HuggingFacePipeline
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
import torch
from huggingface_hub import login
import transformers


app = Flask(__name__)

# Initialize RAG pipeline components
# Use your existing pipeline initialization here
# Assuming `qa_chain` function is already defined in the provided code.

model_id = "meta-llama/Llama-3.1-8B-Instruct"
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
token = "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"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_config = AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=True,
    max_new_tokens=2048,
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

query_pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    max_new_tokens=200,
    device_map="auto"
)

llm = HuggingFacePipeline(pipeline=query_pipeline)

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

def qa_chain(query):
    embedding = embedding_model._client.encode(query, prompt_name="s2p_query")
    embedding = embedding.tolist()
    res = retriever.similarity_search_with_score_by_vector(embedding, k=4)
    context = res[0][0].page_content
    
    chain = prompt | llm
    
    result = chain.invoke(
        {
            "context": context,
            "question": query,
        }
    )
    return result

# Define API endpoint
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    try:
        answer = qa_chain(question)
        return jsonify({"question": question, "answer": answer.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
