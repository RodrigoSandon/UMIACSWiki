# I do not claim the total ownership of this code, help with AI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


client = QdrantClient(path="./qdrant")


if client.collection_exists("parentdoc"):
    client.delete_collection("parentdoc")

# reate new collection
client.create_collection(
    collection_name="parentdoc",
    vectors_config=VectorParams(
        size=1024,  # djust this to match your embedding model's dimension
        distance=Distance.COSINE
    )
)


embedding_model = HuggingFaceEmbeddings(
    model_name="dunzhang/stella_en_1.5B_v5",
    model_kwargs={
        "trust_remote_code": True,
    },
)


vector_store = QdrantVectorStore(
    client=client,
    collection_name="parentdoc",
    embedding=embedding_model,
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


def process_files(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    # reate document with metadata
                    doc = Document(
                        page_content=text,
                        metadata={"source": filename}
                    )
                    documents.append(doc)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return documents

# process all files in your directory
directory_path = "path/to/your/files"  # replace with your files directory
documents = process_files(directory_path)

split_docs = []
for doc in documents:
    splits = text_splitter.split_documents([doc])
    split_docs.extend(splits)

print(f"Created {len(split_docs)} document chunks from {len(documents)} documents")

vector_store.add_documents(split_docs)

print("Vector database creation complete!")
