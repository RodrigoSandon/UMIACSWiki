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

`run` is interactive, so for running a script you can use:
```
echo "" | /fs/nexus-projects/umiacs-wiki-chatbot/ollama/bin/ollama run llama3.1:8b-instruct-fp16
```

ollama uses gpu 0 by default, so in order to run python scripts use:
```
CUDA_VISIBLE_DEVICES=1 script.py
```
this is necessary as langchain and most libraries seem to just use the first available device with no way to change that