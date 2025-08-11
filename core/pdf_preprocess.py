import fitz  
import re


def extract_text_from_pdf(path):
    """
    Extract full text from a PDF file
    
    Args:
        path (str): Path to the PDF file
    """
    with fitz.open(path) as doc:
        full_text = ""
        for page in doc:
            text = page.get_text()
            full_text += text + "\n"
    return full_text


def remove_figures_tables(text):
    """
    Removes lines starting with 'Figure' or 'Table' (common in scientific papers)

    Args:
        text (str): Text to clean
    """
    lines = text.split('\n')
    cleaned = [line for line in lines if not re.match(r'^(Figure|Table)\s*\d+', line)]
    return '\n'.join(cleaned)


def remove_page_numbers(text):
    """
    Removes isolated page numbers (common at bottom of pages)

    Args:
        text (str): Text to clean
    """
    return re.sub(r'\n\d+\n', '\n', text)


def remove_headers_footers(text):
    """
    Removes repeated headers/footers by detecting frequent lines

    Args:
        text (str): Text to clean
    """
    lines = text.split('\n')
    freq = {}
    for line in lines:
        freq[line] = freq.get(line, 0) + 1
    threshold = len(lines) // 10  # adjust as needed
    cleaned = [line for line in lines if freq[line] < threshold]
    return '\n'.join(cleaned)


def clean_text(text):
    """
    Basic cleaning: removes multiple newlines, references, etc

    Args:
        text (str): Text to clean
    """
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'References.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def full_clean(path):
    """
    Full cleaning pipeline: cleans text, removes figures/tables, 
    page numbers, headers/footers

    Args:
        path (str): Path to the PDF file
    """
    text = extract_text_from_pdf(path)
    text = clean_text(text)
    text = remove_figures_tables(text)
    text = remove_page_numbers(text)
    text = remove_headers_footers(text)
    return text


def split_text(text, max_words):
    """
    Split text into chunks of semantically related sentences, each with up to max_words words.

    Args:
        text (str): Text to split
        max_words (int): Maximum number of words per chunk
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks