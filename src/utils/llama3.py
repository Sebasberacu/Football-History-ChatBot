from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class Llama3:
    def __init__(self):
        self.llm = None
        self.prompt = None
        self.chain = None
        self.outputParser = None

    def start(self):
        self.llm = Ollama(model="llama3")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a world-class football historian, knowledgeable about both male and female football since its inception. "
             "Give detailed answers about football, providing historical context and facts. "
             "Never make use of the second person (e.g. 'you', 'your', etc.). "
             "Always be kind and attentive, and offer to answer more questions."),
            ("user", "{question}"),
            ("assistant", "{context}")
        ])
        self.outputParser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.outputParser

    def ask(self, question, context):
        response = self.chain.invoke({
            "context": f"Use this information from important football history books as context to answer the question: {context}. Please stick to this context when answering the question.",
            "question": f"The question from the user is: {question}"
            })
        return response
        