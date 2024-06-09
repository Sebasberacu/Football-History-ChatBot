# **RAG System Project**
**Escuela de Ingeniería en Computación**

**IC6200 - Inteligencia Artificial**

**Estudiantes:**
Gerald Núñez Chavarría - 2021023226
Sebastián Arroniz Rojas - 2021108521
Sebastián Bermúdez Acuña - 2021110666

**Profesor:**
Kenneth Obando Rodríguez

**Fecha de entrega:**
2024-06-09

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

Texto.

## Fine Tuning 

Texto.

## Pruebas Finales 

Texto.

## Análisis de Resultados

Texto.

