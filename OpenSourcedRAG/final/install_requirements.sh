#!/bin/bash

conda install -c conda-forge transformers -y
conda install -c conda-forge langchain -y
conda install -c conda-forge qdrant-client -y
conda install -c conda-forge beautifulsoup4 -y
conda install -c conda-forge pickle5 -y
conda install -c conda-forge torch -y
conda install -c conda-forge python-dotenv -y

echo "All required packages have been installed in the 'my_rag_env' environment."