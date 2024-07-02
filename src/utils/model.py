from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class Model:
    def __init__(self):
        self.llm = None
        self.prompt = None
        self.chain = None
        self.outputParser = None

    def start(self):
        self.llm = Ollama(model="llama3") # Load the Llama3 model
        self.prompt = ChatPromptTemplate.from_messages([ # Chat prompt template
            ("system", 
            "You are a world-class football historian, knowledgeable about football since its inception. "
            "Give detailed answers about football, providing historical context and facts. "
            "Be kind and attentive, and offer to answer more questions. "
            "Incorporate the provided information seamlessly into your responses without directly referencing it."),
            ("user", "{question}"),
            ("assistant", "{context}")
        ])
        self.outputParser = StrOutputParser() # Parse the output as a string
        self.chain = self.prompt | self.llm | self.outputParser # Create the chain

    def ask(self, question, context):
        response = self.chain.invoke({
            "context": f"Relevant historical information: {context}",
            "question": f"The question from the user is: {question}"
        })
        return response
