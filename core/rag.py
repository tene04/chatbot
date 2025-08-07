import torch
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
import faiss
import numpy as np

LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DOCUMENTS = [
    "Los transformers usan atención multi-cabeza para procesar secuencias.",
    "Mistral-7B es un modelo de lenguaje eficiente y de código abierto.",
    "RAG combina recuperación de información con generación de texto.",
]

def load_models():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
    llm = LLM(
        model=LLM_MODEL,
        quantization="awq",
        tensor_parallel_size=1,
    )
    sampling_params = SamplingParams(temperature=0.7, max_tokens=200)
    embeddings = embedding_model.encode(DOCUMENTS, convert_to_tensor=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.cpu().numpy())
    return llm, embedding_model, index, sampling_params

def retrieve_context(query, embedding_model, index, documents, k=2):
    query_embed = embedding_model.encode(query, convert_to_tensor=True)
    distances, indices = index.search(query_embed.cpu().numpy(), k)
    return "\n".join([documents[i] for i in indices[0]])

def generate_response(query, context, llm, sampling_params):
    prompt = f"""Responde basándote en el contexto:
Contexto: {context}
Pregunta: {query}
Respuesta:"""
    outputs = llm.generate(prompt, sampling_params)
    return outputs[0].text.split("Respuesta:")[-1].strip()