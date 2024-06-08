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

## Instalación de Ollama

OLLAMA (Open-source Library for AI Models and Applications) permite acceder a una amplia variedad de modelos y lenguajes de IA de código abierto de manera offline. Se debe instalar `ollama` de manera local. Para esto, puede ingresar al [sitio web de ollama](https://ollama.com/download) dónde puede seleccionar la versión correspondiente al sistema operativo. 

## Extracción de Texto

Parte fundamental del proyecto es extraer texto de archivos pdf. Para esto se va a utilizar la librería de python `fitz` (PyMuPDF). Se abrirá un pdf especifico y se extraerá por páginas. A continuación puede observar el código:

```python
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
```

### Prueba

Para observar qué esta funcionando, podemos llamarla y observar el largo del texto extraído:

```python
pdf_path = "docs/testing/three-little-pigs-story.pdf"
pdf_text = extract_text_from_pdf(pdf_path)
print(len(pdf_text))
```

El resultado es: 7084.  

