import streamlit as st
from core.rag import retrieve_context, generate_response, load_models, DOCUMENTS

llm, embedding_model, index, sampling_params = load_models()

st.title("🚀 Chatbot Avanzado (RAG + Mistral-7B)")
user_query = st.text_input("Pregunta:")

if user_query:
    with st.spinner("Buscando contexto..."):
        context = retrieve_context(user_query, embedding_model, index, DOCUMENTS)
        st.write(f"📚 **Contexto relevante:**\n{context}")
    
    with st.spinner("Generando respuesta..."):
        response = generate_response(user_query, context, llm, sampling_params)
        st.write(f"🤖 **Respuesta:**\n{response}")