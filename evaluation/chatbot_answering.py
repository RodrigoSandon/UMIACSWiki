# I do not claim the total ownership of this code, help with AI

"""
This code implements a chatbot answering system. 

The system first loads a pre-existing vector store and document store containing reference 
materials. When processing questions, it converts them into vector embeddings, searches for relevant 
context using similarity search, and then uses the retrieved context along with the question to generate 
appropriate answers. 

The system uses a prompt template that instructs the model to 
provide direct answers. 

The code uses transformers, langchain, and Qdrant for
 vector storage, and runs on GPU if available. 
"""
import json
import os
import torch
from transformers import AutoTokenizer, AutoConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from OurParentDocumentRetriever import OurParentDocumentRetriever
from langchain_community.llms import HuggingFacePipeline
from dotenv import load_dotenv
import warnings
import logging
import pickle as pkl
import transformers

load_dotenv()

warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR)

# huggingFace model and tokenizer setup
model_id = "meta-llama/Llama-3.1-8B-Instruct"
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
token = os.getenv("llama_instruct_token")

model_config = AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=True,
    token=token,
    quiet=True
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    token=token,
    torch_dtype=torch.float16
)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=token,
    trust_remote_code=True
)

query_pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=400
)
query_pipeline.model.to(device)

# wrap the model pipeline for LangChain
llm = HuggingFacePipeline(pipeline=query_pipeline)

# load the embedding model
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

# load the document store
with open("parentdoc.pkl", "rb") as f:
    docstore = pkl.load(f)

# initialize the retriever
retriever = OurParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=docstore
)

# define the QA chain function
def qa_chain(question):
    # encode the question
    embedding = embedding_model._client.encode(question, prompt_name="s2p_query")
    embedding = embedding.tolist()
    
    res = retriever.similarity_search_with_score_by_vector(embedding, k=3)
    
    # use the top 2 documents for context
    context = "\n\n".join([doc.page_content for doc, score in res[:2]])
    
    # define the prompt template
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
    
    # format the prompt with context and question
    prompt_text = prompt_template.format(context=context, question=question)
    
    # generate the answer
    result = llm(prompt_text)
    
    # extract the answer
    if "Answer:" in result:
        answer_start = result.index("Answer:") + len("Answer:")
        answer = result[answer_start:].strip().split('\n')[0]
    else:
        answer = result.strip().split('\n')[0]
    
    return answer

def main():
    # load the JSON file
    with open('UMIACSQuestions.json', 'r') as f:
        data = json.load(f)

    # open the output JSON file in write mode
    with open('UMIACS_QA_ChatbotAnswers.json', 'w') as f:
        f.write('[\n')  # Start the JSON array

        # for each item in the JSON file, process and write after updating all QAs
        for page_number, item in enumerate(data, start=1):
            # process each QA in the item
            qas = item.get('Q/A', [])
            for qa_index, qa in enumerate(qas, start=1):
                question = qa.get('question')
                if question:
                    print(f"processing page {page_number}, question {qa_index}: {question}")
                    new_answer = qa_chain(question)
                    qa['answer'] = new_answer  # replace the original answer

            # write the updated item to the JSON file
            json.dump(item, f, indent=2)
            if page_number < len(data):
                f.write(',\n')  # add a comma between items
            else:
                f.write('\n')  # for the last item, just add a newline
            f.flush()

        f.write(']')

if __name__ == '__main__':
    main()
