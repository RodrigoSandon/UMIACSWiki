import os

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def get_llama_token():
    return read_file('env/llama_token.txt')

def get_qdrant_api_key():
    return read_file('env/qdrant_api_key.txt')

def get_qdrant_endpoint_url():
    return read_file('env/qdrant_endpoint_url.txt')

def get_local_folder_path():
    return '/Users/rodrigosandon/Documents/CMSC/473/UMIACSWiki/dataset/raw_html'

def get_collection_name():
    return "umiacs_wiki"
