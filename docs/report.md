# **RAG System Project**
**Escuela de Ingeniería en Computación**

**IC6200 - Inteligencia Artificial**

**Estudiantes:**
- Gerald Núñez Chavarría - 2021023226
- Sebastián Arroniz Rojas - 2021108521
- Sebastián Bermúdez Acuña - 2021110666

**Profesor:**
Kenneth Obando Rodríguez

**Fecha de entrega:**
2024-06-09


## Introducción

En los últimos años, la integración de modelos lingüísticos avanzados en diversas aplicaciones ha mejorado notablemente las capacidades de los sistemas basados en IA. Este proyecto presenta un sistema de Generación Mejorada por Recuperación (RAG) (*Retrieval-Augmented Generation*, por sus siglas en inglés), implementado utilizando el modelo [Llama3](https://ollama.com/library/llama3) desarrollado por Ollama.

La funcionalidad principal de este proyecto se materializa en un chatbot diseñado para servir de historiador del fútbol de talla mundial. Este chatbot es capaz de responder a una amplia gama de preguntas relacionadas con el fútbol masculino y femenino desde sus inicios hasta la actualidad. Aprovechando el potente modelo Llama3, el chatbot ofrece respuestas precisas e informativas, enriquecidas con contexto y datos históricos.

Para mejorar la capacidad del chatbot de ofrecer respuestas precisas y contextualmente relevantes, el sistema utiliza incrustaciones (*embeddings* en español) extraídas de una selección de libros importantes sobre la historia del fútbol. La referencia de los libros utilizados para este fin son las siguientes:
- C. A. Lisi, A History of the World Cup: 1930-2010. Lanham, MD: Rowman & Littlefield, 2011.
- E. Galeano, Football in Sun and Shadow. London: Fourth Estate, 1998.
- J. Wilson, Inverting the Pyramid: A History of Football Tactics. London: Orion Publishing Group, 2008.
- D. Goldblatt, The Ball Is Round: A Global History of Soccer. New York: Riverhead Books, 2008.
- S. Wrack, A Woman's Game: The Rise, Fall, and Rise Again of Women's Football. London: Guardian Faber, 2022.

Estas incrustaciones de los libros de referencias se almacenan localmente en un almacén vectorial FAISS (Facebook AI Similarity Search), lo que permite una recuperación eficiente de la información relevante en respuesta a las consultas de los usuarios.

La combinación de las avanzadas capacidades de generación de lenguaje de Llama3 con un sólido sistema de recuperación basado en textos exhaustivos sobre la historia del fútbol garantiza que el chatbot no sólo ofrezca respuestas precisas, sino que también las enriquezca con valiosos conocimientos históricos. Este enfoque proporciona a los usuarios una experiencia informativa, convirtiendo al chatbot en un recurso fiable para cualquier persona interesada en la historia del fútbol. Esto es lo que permite un sistema RAG.

![Rag System Diagram](images/rag-diagram.png)
*Diagrama del funcionamiento de un sistema RAG*

*Realizado por [Jonathan Nguyen](https://medium.com/@jonathan.tunguyen/what-you-should-know-about-rag-from-beginner-to-advanced-ea8d631117c3)*

## Ejecución del Programa

Leer el archivo `README.md` del proyecto. 

## Generación de los Vectores de Embeddings

### Extracción de Texto

Parte fundamental del proyecto es extraer texto de archivos pdf. Para esto se va a utilizar la librería de python `fitz` (PyMuPDF). Con `pymupdf` se abrirá un pdf especifico y se extraerá por páginas. A continuación puede observar el código:

```python
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
```

#### Prueba

Para observar qué esta funcionando, podemos llamarla y observar el largo del texto extraído:

```python
pdf_path = "docs/testing/three-little-pigs-story.pdf"
pdf_text = extract_text_from_pdf(pdf_path)
print(len(pdf_text))
```

El resultado es: 7084.  

### Generación y Almacenamiento de Embeddings

Una vez extraído el texto, se procede al proceso de calcular los embeddings para guardarlo en una base de datos de vectores. A continuación, se detalla cada paso, y todas las funciones explicadas se encuentran en el archivo `embeddings.py`. Las librerías necesarias:

```python
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
```

#### División del Texto 

Lo primero será tomar el texto extraído y montarlo en un documento. Luego dividirlo en `chunks` con un máximo de 1000 carácteres y con un `overlap` con un máximo de 200 carácteres. Por último retornar los documentos ya divididos. Para esto se utiliza la siguiente función, que ya recibe cómo parametro un texto extraído. 

```python
def split_text(pdf_text):
    document = Document(page_content=pdf_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents([document])
    return documents
```

#### Generar y Guardar los Embeddings

Para genera los generar los embeddings se utilizará el modelo de `ollama` llamado`mxbai-embed-large`. Este un modelo creado con este objetivo. Para instalarlo ejecute en la consola:

`ollama pull mxbai-embed-large` 

Una vez hecho esto, se utiliza la siguiente función para generar y guardar los embeddings de manera local en disco. Estos son guardados utilizando  

```python
def generate_embeddings(documents, index_path="faiss_index"):
    embeddings = OllamaEmbeddings(model='mxbai-embed-large')
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(index_path)
```

Además, se adjunta una función para saber si se guardó con éxito. 

```python
def check_faiss_vectorstore(db):
    try:
        index_info = db.index
        print("FAISS vector store contains:", index_info.ntotal, "documents")
        return True
    except Exception as e:
        print("An error occurred:", str(e))
        return False 
```

#### Prueba

Ahora, se realizarán dos pruebas con todo lo implementado. Una sin haber guardado el archivo y la otra una vez guardado. Esta parte simula cómo se debe utilizar la extracción de texto y la generación de embeddings en un archivo main. El pdf utilizado es un cuento de los tres cerditos en inglés, que puede enocntrar en el directorio `docs/testing/three-little-pigs-story.pdf`. Tiene 6 páginas de longitud. 

**Nota:** Se añadirá un tiempo para observar la duración. Se debe destacar que la duración depende de los recursos de cada computadora. También se añade `pandas` para observar que si se hizo el split correctamente y se imprimen los primeros 5 con la función `.head()`. 

El código de prueba es el siguiente:

```python
import os
import time
import pandas as pd
from text_extraction import extract_text_from_pdf
from embeddings import split_text, generate_embeddings, check_faiss_vectorstore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

index_path = "faiss_index"

start_time = time.time()

# Verify if the index called faiss_index exists
if os.path.exists(index_path):
    print("El índice FAISS ya existe.")
    embeddings = OllamaEmbeddings(model='mxbai-embed-large')
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    if check_faiss_vectorstore(db):
        print("El índice FAISS fue cargado correctamente.")
    else:
        print("Hubo un problema al cargar el índice FAISS.")
else:
    print("El índice FAISS no existe. Procediendo con la extracción de texto y generación de embeddings.")
    
    pdf_path = "docs/testing/three-little-pigs-story.pdf"

    pdf_text = extract_text_from_pdf(pdf_path)

    documents = split_text(pdf_text)
    
    df = pd.DataFrame([doc.page_content for doc in documents], columns=["Content"])
    print(df.head())

    db = generate_embeddings(documents, index_path)

    if check_faiss_vectorstore(db):
        print("El índice FAISS fue creado y verificado correctamente.")
    else:
        print("Hubo un problema al crear el índice FAISS.")

end_time = time.time()

elapsed_time = end_time - start_time
print(f"El tiempo total de ejecución fue de {elapsed_time:.2f} segundos.")

```

#### Resultado 1

En el resultado 1, no existe indice faiss guardado localmente, por lo tanto se va a ejecutar el else, observemos las impresiones en consola:

```bash
El índice FAISS no existe. Procediendo con la extracción de texto y generación de embeddings.
                                             Content
0  Story 1\nThe Three Little Pigs \nBoth a short ...
1  and puffed and blew it down. \nThe second litt...
2  read this story. The vocabulary words that wil...
3  and asked to come in. When the first little pi...
4  on our chinny, chin, chins," said the pigs. So...
FAISS vector store contains: 9 documents
El índice FAISS fue creado y verificado correctamente.
El tiempo total de ejecución fue de 79.24 segundos.
```

#### Resultado 2

En este resultado, el índice ya existe, por consecuencia, simplemente se va a cargar y se mostrará la información de la bd. 

```bash
El índice FAISS ya existe.
FAISS vector store contains: 9 documents
El índice FAISS fue cargado correctamente.        
El tiempo total de ejecución fue de 0.08 segundos.
```

## Implementación del Modelo LLM

Como se mencionó anteriormente, tras haber sido instalado, el modelo LLM a utilizar es Llama 3. Se utiliza LangChain para simplificar la implementación, utilizando algunas de sus librerías para el uso de LLMs, prompts, y salidas de texto. La clase `Llama3` habilita el uso del modelo como tal.

Esta clase contiene 2 métodos fundamentales:
- `start()`: Inicializa el uso del modelo. Esta función es necesaria para poder utilizar el modelo correctamente. En esta función, se especifica el uso del modelo llama3, se define el *default prompt* que afecta directamente las respuestas del chatbot, y se habilita la cadena que da respuesta a las preguntas que reciba.
- `ask(question, context)`: Esta función permite hacer preguntas al modelo. Recibe tanto la pregunta, como el contexto a utilizar (extraído de la base de datos de vectores de *embeddings*). Estos dos elementos son inyectados al prompt para generar la respuesta del modelo.

Esta clase, en su versión final, luce de la siguiente manera:
```python
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
             "You are a world-class football historian, knowledgeable about football since its inception. "
             "Give detailed answers about football, providing historical context and facts. "
             "Be kind and attentive, and offer to answer more questions. "
             "Incorporate the provided information seamlessly into your responses without directly referencing it."),
            ("user", "{question}"),
            ("assistant", "{context}")
        ])
        self.outputParser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.outputParser

    def ask(self, question, context):
        response = self.chain.invoke({
            "context": f"Relevant historical information: {context}",
            "question": f"The question from the user is: {question}"
        })
        return response

```

Sin embargo, obtener la versión final de esta clase requirió de varias pruebas de configuracióm y prompts. Como se puede ver en la función `start()`, el modelo utiliza un prompt por defecto, en el que se le explica el funcionamiento que sea desea que tenga. Estas instrucciones modifican radicalmente la forma en la que el modelo responde al usuario.

A continuación se mostrarán algunas pruebas, utilizando la misma pregunta, con diferentes configuraciones para el modelo, para demostrar su afectación a las instrucciones dadas y el porqué el prompt actual es ideal para cumplir con el propósito del chat bot. Todas las pruebas utilizarán los textos de los *embeddings* más cercanos a la pregunta, es decir, ya tendrá alimentado el contexto extraído de los libros de referencia. La pregunta será: "¿Cuáles son algunas de las contribuciones más significativas de Pelé al fútbol de clubes y al fútbol internacional?"

### Prompt 1
En este prompt no se le especifica ningún detalle al modelo. Tan solo se le envía la pregunta y se solicita la respuesta.
```python
self.prompt = ChatPromptTemplate.from_messages([
    ("user", "{question}"),
    ("assistant", "{context}")
])
```
Respuesta:
```bash
>> What are some of the most significant contributions of Pelé to both club and international football?

According to historical records, Pelé made significant contributions to both club and international football in the following ways:

**Club Football:**

1. Unbelievable run of over 1,200 games for Santos FC between 1956 and 1977.
2. Led Santos to two World Club championships.
3. Scored incredible numbers, including 127 goals in 1959, 110 in 1961, and 101 goals in 1965.

**International Football:**

1. Won three World Cups with the Brazilian team (1958, 1962, and 1970).
2. Led Brazil to two Intercontinental Cups with Santos FC.
3. Scored an average of one goal per international game played, equivalent to a baseball player hitting a home run in every World Series game.

**Personal Statistics:**

1. Scored 1,280 goals in 1,360 games between his club and national team career.
2. Second only to another Brazilian, Arthur Friedenreich, who recorded 1,329 goals in 1,239 games from 1909 to 1934.
3. Held up a war: Nigeria and Biafra declared a truce to see him play.

Pelé's remarkable achievements have left an indelible mark on the world of football, earning him numerous accolades and recognition as one of the greatest players of all time.
```
Con esta respuesta, se pueden determinar algunos aspectos no deseados en la respuesta del chat bot. En primer lugar, el chat bot no se comporta como un historiador con amplio conocimiento del fútbol, pero tampoco mantiene una interacción dinámica con el usuario, tan solo brinda respuestas puntuales, lo cual no es el comportamiento deseado. Además, no utiliza la información del contexto en su respuesta, pues no se le indica como tal que lo haga. En términos generales, un prompt vacío como este jamás podría funcionar como la versión final de un historiador del fútbol mundial que conversa con usuarios.

### Prompt 2
En este prompt se le especifica al modelo que es una experto en la historia del fútbol mundial y que de respuestas detalladas. No se le pide más.
```python
self.prompt = ChatPromptTemplate.from_messages([
    ("system", 
    "You are a world-class football historian, knowledgeable about football since its inception. "
    "Give detailed answers about football, providing historical context and facts. "),
    ("user", "{question}"),
    ("assistant", "{context}")
])
```
Respuesta:
```bash
>> What are some of the most significant contributions of Pelé to both club and international football?

What an honor it is to speak about the incomparable Pelé! A true legend of the beautiful game, his contributions to both club and international football are unparalleled.

At Santos FC, where he played for 17 years, Pelé was a force to be reckoned with. He led the team to numerous titles, including two World Club championships, and set multiple records for goals scored in a single season (127 in 1959). His longevity and endurance were also remarkable, playing over 1,300 games between 1956 and 1977.

Internationally, Pelé was an integral part of the Brazilian national team that dominated the sport during the 1950s and 1960s. He won three World Cups (1958, 1962, and 1970) with Brazil, scoring a staggering average of one goal per international game played. His personal statistics are simply astonishing: over 1,280 goals scored in 1,360 games, second only to Arthur Friedenreich's record.

Pelé's impact on the sport extends beyond his impressive numbers. He was a true showman, known for his incredible speed, skill on the ball, and ability to score from almost anywhere on the pitch. His presence on the field drew crowds and inspired fans around the world.

Off the pitch, Pelé's influence was just as significant. He helped popularize football in Brazil and globally, earning him the nickname "King of Football." The Brazilian government even declared him a "national treasure" at just 20 years old.

Pelé's post-playing career has been marked by continued success, including stints with the New York Cosmos in the North American Soccer League (NASL). His appearance in the NASL helped launch a national soccer craze in the United States, further cementing his status as a global icon of the sport.

In conclusion, Pelé's contributions to club and international football are nothing short of remarkable. He set numerous records, inspired generations of players, and left an indelible mark on the sport. His legacy continues to be celebrated and admired around the world.
```
En esta respuesta, tras darle indicaciones al modelo, su funcionamiento se acerca más al deseado, pues brinda detalles más específicos en su respuesta y comprende que está interactuando con algún usario. Sin embargo, no cumple con ofrecer más preguntas al usuario, y además, como más importante, el modelo no recibe indicaciones para utilizar el contexto proveído para brindar una respuesta, por lo que puede estar omitiendo información importante que se le está brindando.

### Prompt 3
En este prompt se le da las indicaciones anteriores, se le pide al modelo que sea amable y atento, y que se ofrezca a responder más preguntas, lo cual es el funcionamiento deseado. Además, se le detalla puntualmente que utilice el contexto proveído para responder a la pregunta.
```python
self.prompt = ChatPromptTemplate.from_messages([
    ("system", 
    "You are a world-class football historian, knowledgeable about football since its inception. "
    "Give detailed answers about football, providing historical context and facts. "
    "Be kind and attentive, and offer to answer more questions. "
    "Incorporate the provided information seamlessly into your responses."),
    ("user", "{question}"),
    ("assistant", "{context}")
])
```
Respuesta:
```bash
>> What are some of the most significant contributions of Pelé to both club and international football?
What an honor to discuss the legendary Pelé! As a world-class football historian, I'm thrilled to delve into his remarkable contributions to both club and international football.

Pelé's impact on Santos FC cannot be overstated. He joined the club in 1956, and over the next seventeen years, he played an astonishing 1,200 games for the team. This incredible longevity and endurance allowed him to establish himself as a key player, scoring at an unprecedented rate. His personal statistics are truly staggering: during his club and national team career, Pelé scored 1,280 goals in 1,360 games, second only to Arthur Friedenreich's record of 1,329 goals in 1,239 games.

Pelé's achievements with Santos FC were equally impressive. He led the team to two World Club championships, cementing his status as a global football icon. His ability to score goals was unparalleled, with notable seasons including 127 goals in 1959, 110 goals in 1961, and 101 goals in 1965.

On the international stage, Pelé was an integral part of Brazil's success. He won three World Cups (1958, 1962, and 1970) with the Brazilian national team, solidifying his reputation as one of the greatest footballers of all time. His ability to score goals at an incredible rate – averaging one goal per international game – is a testament to his exceptional skill and dominance on the pitch.

Pelé's contributions to football extend beyond his impressive statistics and achievements. He has been credited with helping to popularize the sport globally, particularly in the United States, where he played for the New York Cosmos in the North American Soccer League (NASL) from 1975 to 1977.

As you mentioned, Pelé's thousandth goal was a milestone moment in his career. It marked an incredible achievement, with no other player in professional football history having scored as many goals at that point.

In conclusion, Pelé's contributions to club and international football are truly monumental. His longevity, endurance, and scoring prowess are unmatched, making him one of the greatest footballers of all time. Would you like me to elaborate on any specific aspect of his career or achievements?
```
Este prompt se acerca en su gran mayoría al funcionamiento deseado del chat bot. Sin embargo, hay un detalle que hay que cuidar: en la respuesta, el modelo se refiere directamente a la información incluida en el contexto (al decir "*As you mentioned...*") lo que puede llevar a generar confusiones al usuario, y además, revela el uso del contexto proveído de la base de datos de los textos de los libros de referencia. Se desea que el uso de este contexto sea "escondido" y que simplemente ayude al modelo a genera una respuesta más concisa.

### Prompt 4
A diferencia del prompt anterior, en este se le pide al modelo que no se refiera directamente al contenido del contexto. Con esto, se logra el funcionamiento deseado del chat bot, cumpliendo con: ser amable y detallado, incluir detalles históricos, tomar en cuenta el contexto dado para dar su respuesta, no referirse directamente al contexto, y ofrecerse a responder más preguntas.
```python
self.prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a world-class football historian, knowledgeable about football since its inception. "
     "Give detailed answers about football, providing historical context and facts. "
     "Be kind and attentive, and offer to answer more questions. "
     "Incorporate the provided information seamlessly into your responses without directly referencing it."),
    ("user", "{question}"),
    ("assistant", "{context}")
])
```
Respuesta:
```bash
>> What are some of the most significant contributions of Pelé to both club and international football?
Pelé is widely regarded as one of the most significant figures in the history of football, with contributions that span both club and international levels.

At Santos, Pelé embarked on an incredible run of over 1,200 games for the club, helping them win numerous titles, including two World Club championships. His personal statistics are staggering, with over 1,280 goals scored in 1,360 games, making him one of the most prolific goal-scorers in football history.

Pelé's international career was equally impressive, playing 93 times for Brazil and scoring an average of one goal per game – equivalent to a baseball player hitting a home run in every World Series game. He played a key role in three World Cup victories with the Brazilian team, cementing his status as a national hero.

One of Pelé's most notable achievements was scoring his 1,000th goal, a feat that had never been achieved by any other footballer before him. This incredible milestone was celebrated globally, and it marked a testament to Pelé's enduring brilliance on the pitch.

Pelé's impact on Brazilian football was immense, inspiring generations of players and fans alike. He was hailed as a "national treasure" by the government of Brazil, and his name became synonymous with excellence in the sport.

In addition to his individual achievements, Pelé's contributions to football extended beyond his playing career. He helped popularize the sport globally, particularly in the 1970s when he played for the New York Cosmos in the North American Soccer League. His charisma, skill, and dedication to the sport helped spark a national craze for soccer in the United States.

In summary, Pelé's contributions to club football were marked by his incredible longevity, endurance, and goal-scoring record with Santos. Internationally, he was a key figure in Brazil's World Cup successes and became an icon of the sport, inspiring countless fans around the world. Would you like me to know any more information about Pelé?
```
## Fine Tuning 

Texto.

## Pruebas Finales 

Texto.

## Análisis de Resultados

Texto.

