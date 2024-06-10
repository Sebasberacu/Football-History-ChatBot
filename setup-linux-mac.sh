ollama pull llama3
ollama pull mxbai-embed-large

python -m venv .env

source .env/bin/activate

pip install -r requirements.txt

python -m ipykernel install --user --name=venv

huggingface-cli login