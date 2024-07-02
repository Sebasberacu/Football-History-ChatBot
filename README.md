# Football History ChatBot <!-- omit in toc -->

- [Introduction](#introduction)
- [Get Started](#get-started)
  - [Models](#models)
  - [Python Virtual Environment](#python-virtual-environment)
    - [New Dependencies](#new-dependencies)
  - [Hugging Face CLI Login](#hugging-face-cli-login)
  - [Executing The ChatBot](#executing-the-chatbot)

## Introduction
This project corresponds to a chatbot specialized in world football history. Using natural language processing and information retrieval, the system combines an LLM model, specifically Ollama's Llama3, with a RAG (Retrieval-Augmented Generation) approach to provide contextually rich and accurate responses. Embeddings of relevant historical documents have been created and stored in a FAISS vector database, allowing the model to quickly access detailed and reliable information. The chatbot delivers informative responses and acts as an assistant for those interested.

## Get Started

To be able to run this project, you need to download and install [Ollama](https://ollama.com/download). 

Also, you need to go to the [Llama3 model](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and request access to it.

Once you have installed Ollama and you have created the Acces Token in Hugging Face, you can execute the setup script for your specific machine:

* **Windows:** Execute the script `setup-windows.bat`.
* **Linux or Mac:** Execute the script `setup-linux-mac.sh`.

You can also follow the following instructions.

### Models

1. **Pull the Llama3 model**

```bash
ollama pull llama3
```

2. **Pull the Large Embedding Model**

```bash
ollama pull mxbai-embed-large
```

### Python Virtual Environment

To run this project for the first time you have to create a python virtual environment:

1. **Create the virtual environment**

```bash
python -m venv .env
```

2. **Activate the virtual environment**    

**For Windows:**

```powershell
.env/Scripts/activate
```

**For Mac or Linux:**

```bash
source .env/bin/activate
```

3. **Install the dependencies**

```bash
pip install -r requirements.txt
```

4. **Setup the ipynb kernel**

```bash
python -m ipykernel install --user --name=venv
```

5. **To deactivate the virtual environment**

```bash
deactivate
```

#### New Dependencies

If you are using the virtual environment and you install a new package with `pip install`, execute the command 

```bash
pip freeze > requirements.txt
```

so you can update the requirements list.

### Hugging Face CLI Login

You need to be logged in Hugging Face in the terminal to execute the notebook `./src/rag_football_chatbot.ipynb`.

1. **Login from CLI**

```bash
huggingface-cli login
```

2. **Enter the access token**

### Executing The ChatBot

1. Activate the python virtual environment if it is disabled.
2. Execute the file `src/main.py`.
3. The bot is going to prompt you a message.
4. Enter your message and hit enter.
5. The bot is going to generate a response for you.