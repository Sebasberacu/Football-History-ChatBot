python -m venv .env

source .env/bin/activate

pip install -r requirements.txt

python -m ipykernel install --user --name=venv