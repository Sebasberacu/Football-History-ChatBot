# RAG Soccer ChatBot <!-- omit in toc -->

- [Get Started](#get-started)
  - [Ollama](#ollama)
  - [Python Virtual Environment](#python-virtual-environment)
    - [New Dependencies](#new-dependencies)


## Get Started

### Ollama

For this project you need to download [Ollama](https://ollama.com/download).

1. **Pull the Llama3 model:**

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

You can also execute the scripts present in the project for your specific machine. Note that they are `.sh` files, so you have to execute them with a bash terminal.

#### New Dependencies

If you are using the virtual environment and you install a new package with `pip install`, execute the command 

```bash
pip freeze > requirements.txt
```

so you can update the requirements list.