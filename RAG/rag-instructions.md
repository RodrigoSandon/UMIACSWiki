ollama is installed in /fs/nexus-projects/umicas-wiki-chatbot/ollama

`srun --pty --partition=class --qos=medium --gres=gpu:rtxa5000:2 bash` \
2 gpus for rag because 1 for llm, 1 for embedding model. 2 gpus requires `--qos=medium`

set ollama model folder:
```
export OLLAMA_MODELS=/fs/nexus-projects/umiacs-wiki-chatbot/ollama
```

start the server:
```
/fs/nexus-projects/umiacs-wiki-chatbot/ollama/bin/ollama serve &
```

running the model isn't necessary if using langchain because it does it for you.

run model using 
```
/fs/nexus-projects/umiacs-wiki-chatbot/ollama/bin/ollama run modelname
```
modelnames are from https://ollama.com/library/

example:
```
/fs/nexus-projects/umiacs-wiki-chatbot/ollama/bin/ollama run llama3.1:8b-instruct-fp16
```

by default ollama models are quantized so look for the tags that aren't

# i havent tested anything after this


`run` is interactive so if you do need to run it you can use this which will also preload the model or something:
```
echo "" | /fs/nexus-projects/umiacs-wiki-chatbot/ollama/bin/ollama run llama3.1:8b-instruct-fp16
```

ollama uses gpu 0 by default, so to run python scripts on gpu 1 use:
```
CUDA_VISIBLE_DEVICES=1 python script.py
```
HOWEVER ollama only puts the model in vram when a chat request is made, so if you do that after loading the embedding model, ollama will put the model on the other gpu anyways (?). idk what happens if you do it in the other order.


# also here is a run script for what's currently in ragtest2.py
doing all tickets, but will take hours. llm output takes ~15s avg and embedding takes like lets say 5s (less in reality). 1800 tickets * 20s/ticket = 10 hrs estimated
```
#!/bin/bash
#SBATCH --partition=class
#SBATCH --qos=medium
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:rtxa5000:2

source activate dl
export OLLAMA_MODELS=/fs/nexus-projects/umiacs-wiki-chatbot/ollama
/fs/nexus-projects/umiacs-wiki-chatbot/ollama/bin/ollama serve &
cd ~/UMIACSWiki/Custom_RAG
CUDA_VISIBLE_DEVICES=1 python ragtest2.py
```