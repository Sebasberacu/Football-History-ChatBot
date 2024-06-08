from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def splitText(pdfText):
    document = Document(page_content=pdfText)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents([document])
    return documents

def generateEmbeddings(documents, index_path="faiss_index"):
    embeddings = OllamaEmbeddings(model='mxbai-embed-large')
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(index_path)
    return db

def checkFaissVectorstore(db):
    try:
        index_info = db.index
        print("FAISS vector store contains:", index_info.ntotal, "documents")
        return True
    except Exception as e:
        print("An error occurred:", str(e))
        return False 