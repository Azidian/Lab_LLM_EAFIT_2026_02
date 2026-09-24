import streamlit as st
import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuración de la página
st.set_page_config(
    page_title="Plataforma OCR & LLM con OpenAI",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Extracción OCR de Imágenes y Ampliación con LLM (OpenAI)")
st.markdown("Sube una imagen, extrae su texto mediante OCR, amplíalo con modelos GPT seleccionando el tono (Formal o Técnico) y analiza métricas avanzadas del texto generado.")

# --- BARRA LATERAL: CONFIGURACIÓN Y CREDENCIALES ---
st.sidebar.header("⚙️ Configuración")

# Ingreso de la API Key de OpenAI
api_key_input = st.sidebar.text_input("Ingresa tu API Key de OpenAI", type="password")

if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input

# Selección de Modelos GPT disponibles
model_options = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo"
]
selected_model = st.sidebar.selectbox("Selecciona el Modelo GPT", model_options)

# Selección de Tono de Respuesta
tone_option = st.sidebar.radio("Elige el Tono de la Respuesta", ["Formal", "Técnica"])

# Parámetros avanzados del LLM
st.sidebar.subheader("🎛️ Parámetros del Modelo")
temperature = st.sidebar.slider("Temperatura", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
max_tokens = st.sidebar.slider("Tokens Máximos", min_value=50, max_value=2048, value=800, step=50)
top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
presence_penalty = st.sidebar.slider("Penalización de Presencia", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
frequency_penalty = st.sidebar.slider("Penalización de Frecuencia", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

# --- FUNCIÓN DE CÁLCULO DE MÉTRICAS DEL TEXTO ---
def calculate_text_metrics(original_text, generated_text):
    words = generated_text.split()
    num_words = len(words)
    num_chars = len(generated_text)
    
    # Oraciones aproximadas por puntuación
    sentences = re.split(r'[.!?]+', generated_text)
    num_sentences = max(len([s for s in sentences if s.strip()]), 1)
    
    # Diversidad léxica (Type-Token Ratio)
    ttr = len(set(w.lower() for w in words)) / num_words if num_words > 0 else 0
    
    # Promedio de palabras por oración (Sintaxis)
    avg_words_per_sentence = num_words / num_sentences
    
    # Similitud semántica entre el texto extraído del OCR y el generado por el LLM
    try:
        vectorizer = TfidfVectorizer().fit_transform([original_text, generated_text])
        vectors = vectorizer.toarray()
        semantic_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    except Exception:
        semantic_sim = 0.0
        
    # Estimación heurística de coherencia sintáctica / gramatical basada en longitud y estructura
    coherence_score = min(max(semantic_sim * 1.2 + (0.2 if avg_words_per_sentence > 8 else 0.1), 0.1), 1.0)
    
    return {
        "Palabras": num_words,
        "Caracteres": num_chars,
        "Oraciones": num_sentences,
        "Promedio Palabras/Oración": round(avg_words_per_sentence, 2),
        "Diversidad Léxica (TTR)": round(ttr, 3),
        "Similitud Semántica (OCR vs LLM)": round(semantic_sim, 4),
        "Puntuación de Coherencia": round(coherence_score * 100, 1),
        "Sintaxis / Gramática (Est. Estructural)": "Alta" if avg_words_per_sentence >= 10 else "Moderada"
    }

# --- INTERFAZ PRINCIPAL ---
st.markdown("---")

uploaded_image = st.file_uploader("Sube una imagen con texto (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

extracted_text = ""

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(image, caption="Imagen Cargada", use_container_width=True)
        
    with col_img2:
        st.subheader("📄 Texto Extraído mediante OCR")
        with st.spinner("Procesando imagen con OCR..."):
            try:
                # Extraer texto usando Tesseract OCR
                extracted_text = pytesseract.image_to_string(image)
            except Exception as e:
                st.error(f"Error al ejecutar OCR. Asegúrate de tener Tesseract instalado en el sistema. Detalles: {e}")
                extracted_text = ""
                
        if extracted_text.strip():
            st.text_area("Texto detectado:", extracted_text, height=200)
        else:
            st.warning("No se detectó texto claro en la imagen o el motor OCR requiere asistencia manual.")
            extracted_text = st.text_area("O puedes editar o ingresar el texto base aquí:", "")

# Sección de Ampliación con LLM
if extracted_text.strip():
    st.markdown("---")
    st.header("🤖 Ampliación y Análisis con LLM (OpenAI)")
    
    additional_prompt = st.text_input("Instrucción adicional opcional para el modelo:", "Amplía, estructura y analiza el contenido técnico o conceptual del texto extraído.")
    
    if st.button("Ejecutar Ampliación con GPT"):
        if not api_key_input:
            st.error("Por favor, ingresa tu API Key de OpenAI en la barra lateral.")
        else:
            try:
                client = OpenAI(api_key=api_key_input)
                
                # Construir system prompt según el tono seleccionado
                if tone_option == "Formal":
                    system_prompt = "Eres un asistente experto de redacción ejecutiva. Amplía y explica el texto proporcionado manteniendo un tono estrictamente formal, profesional, claro y corporativo."
                else: # Técnica
                    system_prompt = "Eres un ingeniero y científico experto. Amplía y explica el texto proporcionado adoptando un tono estrictamente técnico, riguroso, analítico y empleando terminología especializada adecuada."
                
                user_content = f"Texto base extraído por OCR:\n{extracted_text}\n\nInstrucción adicional: {additional_prompt}"
                
                with st.spinner(f"Generando respuesta ampliada con {selected_model} (Tono: {tone_option})..."):
                    response = client.chat.completions.create(
                        model=selected_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty
                    )
                    
                    generated_response = response.choices[0].message.content
                    
                    # Guardar en la sesión de Streamlit para persistencia
                    st.session_state["generated_response"] = generated_response
                    st.session_state["extracted_text"] = extracted_text
                    
            except Exception as e:
                st.error(f"Ocurrió un error al conectar con la API de OpenAI: {e}")

# Mostrar resultados y métricas si ya se generó contenido
if "generated_response" in st.session_state:
    st.markdown("### 📝 Respuesta Ampliada del LLM:")
    st.write(st.session_state["generated_response"])
    
    st.markdown("---")
    st.header("📊 Métricas del Texto Generado")
    
    metrics = calculate_text_metrics(st.session_state["extracted_text"], st.session_state["generated_response"])
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Palabras", metrics["Palabras"])
        st.metric("Diversidad Léxica (TTR)", metrics["Diversidad Léxica (TTR)"])
    with col_m2:
        st.metric("Total Oraciones", metrics["Oraciones"])
        st.metric("Similitud Semántica", f"{metrics['Similitud Semántica (OCR vs LLM)'] * 100}%")
    with col_m3:
        st.metric("Promedio Palabras/Oración", metrics["Promedio Palabras/Oración"])
        st.metric("Puntuación Coherencia", f"{metrics['Puntuación de Coherencia']}/100")
    with col_m4:
        st.metric("Caracteres Totales", metrics["Caracteres"])
        st.metric("Sintaxis / Gramática", metrics["Sintaxis / Gramática (Est. Estructural)"])
