import fitz  
import re

def extract_text_from_pdf(path):
    """
    Extract full text from a PDF file
    
    Args:
        path (str): Path to the PDF file
    """
    doc = fitz.open(path)
    full_text = ""
    for page in doc:
        text = page.get_text()
        full_text += text + "\n"
    return full_text

def clean_text(text):
    """
    Basic cleaning: removes multiple newlines, references, etc.

    Args:
        text (str): Text to clean
    """
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'References.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()

def split_text(text, max_words=200):
    """
    Divide text into chunks of maximum `max_words` words
    
    Args:
        text (str): Text to split
        max_words (int): Maximum number of words per chunk
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks
