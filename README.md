# RAG Soccer ChatBot

## Get Started

### Ollama

For this project you need to download [Ollama](https://ollama.com/download).

After you downloaded it, you have to pull 2 models with the following commands:

* Llama3

```bash
ollama pull llama3
```

* Large Embedding Model

```bash
ollama pull mxbai-embed-large
```

### Python Virtual Environment

To run this project for the first time you have to create a python virtual environment. To do this just execute the corresponding script:

* For Windows execute the script: `init-venv-windows.sh`.
* For Mac or Linux execute the script: `init-venv-mac-linux.sh`.

**NOTE:** It's a `sh` file, so it has to be executed with a bash terminal.

#### Deactivation - Activation 

* To deactivate the virtual environment execute the command:

```bash
deactivate
```

To activate the virtual environment execute the following command:

* For Windows execute the command :

```
.env/Scripts/activate
```

* For Mac or Linux execute the command:

```bash
source .env/bin/activate
```

**NOTE:** If you are using a bash terminal to execute the Windows command, you have to add the word `source` at the beginning like this `source .env/Scripts/activate`.

#### New Dependencies

If you are using the virtual environment and you install a new package with `pip install`, execute the command 

```bash
pip freeze > requirements.txt
```

so you can update the requirements list.