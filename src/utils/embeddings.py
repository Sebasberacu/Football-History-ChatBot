from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Split the text into chunks of 1000 characters with an overlap of 200 characters
def splitText(pdfText):
    document = Document(page_content=pdfText)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents([document])
    return documents

# Generates and saves the embeddings of the documents in the FAISS vector store
def generateEmbeddings(documents, embeddingsPath="../faiss_football_documents/"):
    embeddings = OllamaEmbeddings(model='mxbai-embed-large')
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(embeddingsPath)
    return db

# Check if the FAISS vector store exists
def checkFaissVectorstore(db):
    try:
        return True if db.index else False
    except Exception as e:
        print("An error occurred:", str(e))
        return False 