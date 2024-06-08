from utils import *
import sys

# Create a new instance of the Llama3 class
llama3 = Llama3()
llama3.start()


def askChatbot(question):
    """
    Pasos por hacer aquí:
    • Hacer embedding de la pregunta
    • Comparar el embedding de la pregunta con los embeddings de las preguntas en la base de datos para obtener un valor de resultado de la comparación
    • La forma de hacer la comparación es haciendo producto punto (son vectores) entre el embedding de la pregunta y los embeddings de las preguntas en la base de datos
    • Se toman el top 4/5 de los resultados de la comparación y se concatenan con un "\n"
    • Incluir ese string al modelo llama3 (agregar método)
    """
    # Context example 
    context = "Messi is the greatest football player of all time there is no other player like him. Nobody will ever reach him. He has scored more than 700 goals in his career and has won 6 Ballon d'Ors. He has won 10 La Liga titles and 4 Champions Leagues. He has also won the Copa America with Argentina. He is the best player in the world."
    return llama3.ask(question, context)

# Initialize the chatbot
def footballChatBot():
    print("\nWelcome to the football chatbot!\n"+
          "I am a world-class football historian and I am here to talk about anything you want about football.\n\n"+
          "To exit the chatbot, type 'exit' at any time.")
    
    running = True
    while (running):
        question = input("\n>> ")
        if question.lower() == "exit" or question == "e": running = False

        else:
            response = askChatbot(question)
            print(response)
    
    sys.exit(0) # Exit the program

if __name__ == "__main__":
    footballChatBot()