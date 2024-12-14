# UMIACSWiki

Welcome to the **UMIACSWiki** repository! The primary goal of this project was to develop two chatbot systems capable of answering questions based on information from the UMIACS wiki. To achieve this we have:

- Built tools to scrape and extract data from the UMIACS wiki.
- Processed the extracted data into structured formats suitable for training and evaluation.
- Implemented systems that can effectively answer questions using both retrieval-based and model-based methods.
- Established a comprehensive benchmark for evaluating the performance of question-answering systems on wiki-based datasets.

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [RAG](#rag)
  - [Fine-tuning](#fine-tuning)
- [Evaluation](#evaluation)
---

## Prerequisites

Ensure you have the following resources:
- **CPU** >= 4 cores
- **RAM** >= 32 GB
- **Disk** >= 32 GB
- HuggingFace account
- **Clone the repository**:
   ```bash
   git clone https://github.com/RodrigoSandon/UMIACSWiki.git
- Fine-tuning model on LLaMA-8B-instruct available at:
https://drive.google.com/drive/folders/1ymBdxD07nHIs18DXtY0ZOL-b0cshCzzM?usp=drive_link
   
## Getting Started
### Web Scraping 
1. To scrape the UMIACS wiki interact with ```scraping/UMIACSWikiTextScrape.ipynb```
2. Scraped raw text can be found at ```scraping/scrapedText.txt```
3. Qdrant vector database can be found at ```RAG/rag/qdrant/collection/parentdoc/storage.sqlite``` 
### RAG
1. **To run the RAG ui**:
   ```bash
   cd UMIACSWiki/UI 
   pip install -r requirements.txt
   python app.py # on GPU 
   streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501 on cpu # on CPU
   # Chatbot should be availble at http://0.0.0.0:8501/
   ```
2. **To test the RAG pipeline**
   ```bash
   cd UMIACSWiki/UI
   python -m unittest test_app.py
   ```
### Fine-tuning 

1. **To run the fine-tuning and Weibu UI using Llama-factory**

```bash
cd /fs/classhomes/fall2024/cmsc473/c4730027/LLaMA-Factory
CUDA_VISIBLE_DEVICES=0 GRADIO_SHARE=1 GRADIO_SERVER_PORT=7860 llamafactory-cli webui
```
2. **In the terminal, copy the port code then open it in the browser.**
3. **Get the hugging face access tokens and pass the authentication locally**

```bash
huggingface-cli login --add-to-git-credential
```
4. **Load the check-point model(normally named by date-time)**
5. **The chatbot should able to be run locally or using API**

## Evaluation 
1. Question dataset can be found here ```questions/UMIACSQuestions.json``` 

