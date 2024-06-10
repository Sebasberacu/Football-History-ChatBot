import fitz  # PyMuPDF
import os

def extractTextFromPdf(fileName):
    currentDir = os.path.dirname(os.path.abspath(__file__))
    pdfFilePath = os.path.join(currentDir, '..', '..', 'docs', 'knowledge-database', 'documents', fileName)

    try:
        doc = fitz.open(pdfFilePath)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print("An error occurred:", str(e))
        return ""