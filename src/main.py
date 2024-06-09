from utils import *
import sys
import time
import pandas as pd

# Create a new instance of the Llama3 class
llama3 = Llama3()
llama3.start()

# Path to the embeddings folder
embeddingsPath = "faiss_football_documents"

# List of the documents to be used by the RAG System
pdfFiles = [
    "A History of the World Cup 19302010.pdf",                            # C. A. Lisi, A History of the World Cup: 1930-2010. Lanham, MD: Rowman & Littlefield, 2011.
    "Football in Sun and Shadow.pdf",                                     # E. Galeano, Football in Sun and Shadow. London: Fourth Estate, 1998.
    "Inverting the Pyramid The History of Football Tactics.pdf",          # J. Wilson, Inverting the Pyramid: A History of Football Tactics. London: Orion Publishing Group, 2008.
    "The ball is round.pdf",                                              # D. Goldblatt, The Ball is Round: A Global History of Football. London: Penguin, 2006.
    "A Womans Game The Rise, Fall, and Rise Again of Womens Football.pdf" # S. Wrack, A Woman's Game: The Rise, Fall, and Rise Again of Women's Football. London: Guardian Faber, 2022.
]

# Create or load the embeddings database
def createEmbeddings(embeddingsPath):
    if os.path.exists(embeddingsPath):  # If the embeddings db folder exists
        embeddings = OllamaEmbeddings(model='mxbai-embed-large')
        db = FAISS.load_local(embeddingsPath, embeddings, allow_dangerous_deserialization=True)  # Load the index
        return db if checkFaissVectorstore(db) else None
    
    else:  # If the index does not exist, create it
        currentDir = os.path.dirname(os.path.abspath(__file__))
        allDocuments = []
        
        for document in pdfFiles: # Extract the text from the pdf files
            documentPath = os.path.join(currentDir, '..', 'docs', 'knowledge-database', 'documents', document)
            
            documentText = extractTextFromPdf(documentPath)
            splittedDocument = splitText(documentText)
            allDocuments.extend(splittedDocument)
        
        # Print the first 5 documents if wanted
        # df = pd.DataFrame([doc.page_content for doc in allDocuments], columns=["Content"])
        # print(df.head())

        # TODO: Delete time measures and print statements
        print("Proceeding with the embeddings generation.")
        start_time = time.time()
        db = generateEmbeddings(allDocuments, embeddingsPath) # Generate and save the embeddings
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"El tiempo total de ejecución de los embeddings fue de {elapsed_time:.2f} segundos.")

        return db if checkFaissVectorstore(db) else None

# Ask the chatbot a question with a given context
def askChatbot(embeddingsDB, question):
    questionTopSimilarityChunks = embeddingsDB.similarity_search(question, k=5) # Get the top similar chunks to the question from the embeddings database
    context = ""
    for chunk in questionTopSimilarityChunks: # Join the top similar vectors in a single string
        context += chunk.page_content

    return llama3.ask(question, context) # Ask the model with the provided context

# Initialize the chatbot
def footballChatBot():
    embeddingsDB = createEmbeddings(embeddingsPath) # Load or charge the embeddings db
    print("\nWelcome to the football chatbot!\n"+
          "I am a world-class football historian and I am here to talk about anything you want about football.\n\n"+
          "To exit the chatbot, type 'exit' at any time.")
    
    running = True
    while (running):
        question = input("\n>> ")
        if question.lower() == "exit" or question == "e": running = False
        else:
            response = askChatbot(embeddingsDB, question)
            print(response)
    
    sys.exit(0) # Exit the program

if __name__ == "__main__":
    footballChatBot()