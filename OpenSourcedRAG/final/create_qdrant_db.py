from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Initialize Qdrant client
client = QdrantClient(path="./qdrant")

# Delete collection if it exists
if client.collection_exists("parentdoc"):
    client.delete_collection("parentdoc")

# Create new collection
# Note: The size (1024) should match your embedding model's output dimension
client.create_collection(
    collection_name="parentdoc",
    vectors_config=VectorParams(
        size=1024,  # Adjust this to match your embedding model's dimension
        distance=Distance.COSINE
    )
)

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="dunzhang/stella_en_1.5B_v5",
    model_kwargs={
        "trust_remote_code": True,
    },
)

# Create vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name="parentdoc",
    embedding=embedding_model,
)

# Initialize text splitter for chunking documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Function to read files and create documents
def process_files(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    # Create document with metadata
                    doc = Document(
                        page_content=text,
                        metadata={"source": filename}
                    )
                    documents.append(doc)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return documents

# Process all files in your directory
directory_path = "path/to/your/files"  # Replace with your files directory
documents = process_files(directory_path)

# Split documents into chunks
split_docs = []
for doc in documents:
    splits = text_splitter.split_documents([doc])
    split_docs.extend(splits)

print(f"Created {len(split_docs)} document chunks from {len(documents)} documents")

# Add documents to vector store
vector_store.add_documents(split_docs)

print("Vector database creation complete!")
