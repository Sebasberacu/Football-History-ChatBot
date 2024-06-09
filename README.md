# RAG Soccer ChatBot <!-- omit in toc -->

- [Get Started](#get-started)
  - [Models](#models)
  - [Python Virtual Environment](#python-virtual-environment)
    - [New Dependencies](#new-dependencies)


## Get Started

For this project, you need to download and install [Ollama](https://ollama.com/download).

Once you have installed Ollama, you can execute the setup script for your specific machine:

* **Windows:** Execute the script `setup-windows.bat`.
* **Linux or Mac:** Execute the script `setup-linux-mac.sh`.

If you don't want to execute these scripts, you can follow the following instructions.

### Models

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

#### New Dependencies

If you are using the virtual environment and you install a new package with `pip install`, execute the command 

```bash
pip freeze > requirements.txt
```

so you can update the requirements list.