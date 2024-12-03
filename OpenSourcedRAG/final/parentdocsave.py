from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.storage import InMemoryStore
from qdrant_client.http.models import Distance, VectorParams
from OurParentDocumentRetriever import OurParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
import os
import re
from langchain_core.documents import Document

client = QdrantClient(path="./qdrant")

if client.collection_exists("parentdoc"):
    client.delete_collection("parentdoc")

client.create_collection(
    collection_name="parentdoc",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)
# cosine distance in qdrant is actually similarity
# cosine distance in redis is cosine distance (1 - similarity)

store = (
    InMemoryStore()
)  # theres a way to do this using a different database still in redis but idk how

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

child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

retriever = OurParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=store,
    child_splitter=child_splitter,
)


def get_html(file, sep=" "):

    # stolen from https://stackoverflow.com/a/75501596 this sucks
    NON_BREAKING_ELEMENTS = [
        "a",
        "abbr",
        "acronym",
        "audio",
        "b",
        "bdi",
        "bdo",
        "big",
        "button",
        "canvas",
        "cite",
        "code",
        "data",
        "datalist",
        "del",
        "dfn",
        "em",
        "embed",
        "i",
        "iframe",
        "img",
        "input",
        "ins",
        "kbd",
        "label",
        "map",
        "mark",
        "meter",
        "noscript",
        "object",
        "output",
        "picture",
        "progress",
        "q",
        "ruby",
        "s",
        "samp",
        "script",
        "select",
        "slot",
        "small",
        "span",
        "strong",
        "sub",
        "sup",
        "svg",
        "template",
        "textarea",
        "time",
        "u",
        "tt",
        "var",
        "video",
        "wbr",
    ]

    def html_to_text(text, preserve_new_lines=True, strip_tags=["style", "script"]):
        soup = BeautifulSoup(text, "html.parser")
        for element in soup(strip_tags):
            element.extract()
        prev = False
        if preserve_new_lines:
            for element in soup.find_all():
                strings = element.find_all(string=True, recursive=False)
                if strings is not None:
                    strings = [
                        s
                        for s in (None if s.strip() == "" else s for s in strings)
                        if s is not None
                    ]
                strings = strings is not None and strings != []
                strings = True
                if element.name not in NON_BREAKING_ELEMENTS and strings:
                    (
                        element.append("\n")
                        if element.name == "br"
                        else element.append("\n\n")
                    )
        return soup.get_text(separator="")  # close enough

    def replace_newlines(text):
        text = re.sub(r"\n{3}", "\n", text)
        text = re.sub(r"\n{4,}", "\n\n", text)
        return text

    with open(file, "r") as f:
        soup = BeautifulSoup(f, "html.parser")
        iwant = soup.find_all("div", {"id": "content"})
        assert len(iwant) == 1
        text = html_to_text(str(iwant[0]))
        # text = soup.get_text(separator=sep)
        # text = str(iwant[0]) # recursivecharactertextsplitter has html splitting built in so i will try that
        # it sucks nevermind
        text = replace_newlines(text).strip()
        return text  # idk how to do better idk why there's just double spaces sometimes
    return "?"


dataset_path = "../dataset/raw_html"
files = os.listdir(dataset_path)
pages = []

for f in files:
    pages.append(get_html(dataset_path + "/" + f))

docs = []
for i, page in enumerate(pages):
    docs.append(Document(page, metadata={"name": files[i]}))

child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)  # characters not toks

print(len(docs), len(files))

retriever.add_documents(docs, ids=files)
assert len(files) == len(
    list(store.yield_keys())
), f"{len(files)} {len(list(store.yield_keys()))}"

with open("parentdoc.pkl", "wb") as out:
    pickle.dump(store, out, -1)

print("done")
