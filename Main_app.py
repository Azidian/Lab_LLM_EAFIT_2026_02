import streamlit as st
import os
import numpy as np
import pandas as pd
from groq import Groq
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(
    page_title="Plataforma LLM & Procesamiento de Texto (Groq)",
    page_icon="🤖",
    layout="wide"
)

st.title("🚀 Plataforma Interactiva de LLM y Procesamiento de Lenguaje Natural (PLN)")
st.markdown("Explora la generación de texto con **Groq**, visualiza tokens con colores, Bag of Words, matrices de similitud, embeddings y comparativas de temperatura.")

# --- BARRA LATERAL: CONFIGURACIÓN Y API KEY ---
st.sidebar.header("⚙️ Configuración")

# Ingreso de la API Key de Groq
api_key_input = st.sidebar.text_input("Ingresa tu API Key de Groq", type="password")

if api_key_input:
    os.environ["GROQ_API_KEY"] = api_key_input

# Selección de Modelos disponibles en Groq (GPT / Llama / Mixtral / Gemma)
model_options = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]
selected_model = st.sidebar.selectbox("Selecciona el Modelo LLM", model_options)

# Parámetros globales del modelo
st.sidebar.subheader("🎛️ Parámetros del Modelo")
temperature = st.sidebar.slider("Temperatura", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
max_tokens = st.sidebar.slider("Tokens Máximos", min_value=50, max_value=2048, value=512, step=50)
top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

# --- PESTAÑAS PRINCIPALES ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Generación LLM", 
    "⚖️ Comparador de Respuestas",
    "🔤 Tokens & Particiones a Colores", 
    "📦 Bag of Words", 
    "📐 Similitud", 
    "🧬 Embeddings"
])

# ----------------------------------------------------
# TAB 1: GENERACIÓN DE TEXTO CON GROQ
# ----------------------------------------------------
with tab1:
    st.header("Generación de Texto con Modelos Groq")
    
    prompt_text = st.text_area("Escribe tu instrucción (Prompt):", "Explica brevemente qué es la inteligencia artificial.")
    
    if st.button("Generar Respuesta", key="btn_gen"):
        if not api_key_input:
            st.error("Por favor, ingresa tu API Key de Groq en la barra lateral.")
        else:
            try:
                client = Groq(api_key=api_key_input)
                with st.spinner("Generando respuesta..."):
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt_text}],
                        model=selected_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p
                    )
                    response_content = chat_completion.choices[0].message.content
                    
                    st.success("¡Respuesta generada con éxito!")
                    st.markdown("### Resultado:")
                    st.write(response_content)
                    
                    if hasattr(chat_completion, 'usage') and chat_completion.usage:
                        st.info(f"Tokens de entrada: {chat_completion.usage.prompt_tokens} | "
                                f"Tokens de salida: {chat_completion.usage.completion_tokens} | "
                                f"Total de tokens: {chat_completion.usage.total_tokens}")
            except Exception as e:
                st.error(f"Ocurrió un error al conectar con Groq: {e}")

# ----------------------------------------------------
# TAB 2: COMPARADOR DE RESPUESTAS (TEMPERATURA Y PARÁMETROS)
# ----------------------------------------------------
with tab2:
    st.header("⚖️ Comparador de Respuestas por Parámetros")
    st.markdown("Evalúa cómo afecta cambiar la temperatura y las configuraciones en las respuestas del mismo modelo.")
    
    comp_prompt = st.text_input("Pregunta o Prompt a comparar:", "Dame una idea creativa para una startup de tecnología.")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.subheader("Configuración A")
        temp_a = st.slider("Temperatura A", 0.0, 2.0, 0.1, 0.1, key="t_a")
    with col_p2:
        st.subheader("Configuración B")
        temp_b = st.slider("Temperatura B", 0.0, 2.0, 1.5, 0.1, key="t_b")
        
    if st.button("Comparar Respuestas"):
        if not api_key_input:
            st.error("Por favor, ingresa tu API Key de Groq en la barra lateral.")
        else:
            client = Groq(api_key=api_key_input)
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.markdown(f"### Respuesta A (Temp: {temp_a})")
                with st.spinner("Generando A..."):
                    res_a = client.chat.completions.create(
                        messages=[{"role": "user", "content": comp_prompt}],
                        model=selected_model,
                        temperature=temp_a,
                        max_tokens=300
                    )
                    st.write(res_a.choices[0].message.content)
                    
            with col_res2:
                st.markdown(f"### Respuesta B (Temp: {temp_b})")
                with st.spinner("Generando B..."):
                    res_b = client.chat.completions.create(
                        messages=[{"role": "user", "content": comp_prompt}],
                        model=selected_model,
                        temperature=temp_b,
                        max_tokens=300
                    )
                    st.write(res_b.choices[0].message.content)

# ----------------------------------------------------
# TAB 3: TOKENS & PARTICIONES CON COLOR
# ----------------------------------------------------
with tab3:
    st.header("🔤 Tokens y Particiones con Colores")
    st.markdown("Visualiza cómo diferentes métodos (Palabras, Caracteres y Subwords/BPE simulados) dividen un texto, resaltando cada partición con colores dinámicos.")
    
    token_text = st.text_area("Texto para analizar particiones:", "La inteligencia artificial avanza rápidamente en plataformas de procesamiento de lenguaje natural.")
    
    # Selector de método de tokenización
    tok_method = st.selectbox("Selecciona el Método de Tokenización:", ["Por Palabras (Word-level)", "Por Caracteres (Char-level)", "Subwords / BPE (Simulado)"])
    
    if token_text:
        # Generar particiones según el método
        if tok_method == "Por Palabras (Word-level)":
            partitions = token_text.split()
        elif tok_method == "Por Caracteres (Char-level)":
            partitions = list(token_text)
        else:  # Subwords simulado (fragmentación de palabras largas o prefijos comunes)
            raw_words = token_text.split()
            partitions = []
            for w in raw_words:
                if len(w) > 5:
                    partitions.append(w[:3])
                    partitions.append(w[3:] + " ")
                else:
                    partitions.append(w + " ")
                    
        st.markdown(f"### Visualización de Particiones ({tok_method})")
        
        # Paleta de colores estilo HTML/Markdown para resaltar
        colors = ["#ffcdd2", "#c8e6c9", "#bbdefb", "#fff9c4", "#e1bee7", "#ffe0b2", "#b2dfdb"]
        
        html_output = "<div style='line-height: 2.5; font-size: 18px;'>"
        token_table_data = []
        
        for i, part in enumerate(partitions):
            color = colors[i % len(colors)]
            token_id = abs(hash(part)) % 50000
            # HTML para mostrar la partición con fondo de color tipo etiqueta
            html_output += f"<span style='background-color: {color}; padding: 4px 8px; margin: 2px; border-radius: 4px; border: 1px solid #ccc; color: #000;'><b>{part}</b><sub>[{token_id}]</sub></span> "
            token_table_data.append({"Índice": i+1, "Partición / Token": part, "Token ID": token_id})
            
        html_output += "</div>"
        st.markdown(html_output, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Tabla de Tokens e IDs Generados")
        st.dataframe(pd.DataFrame(token_table_data), use_container_width=True)

# ----------------------------------------------------
# TAB 4: BAG OF WORDS (BOW)
# ----------------------------------------------------
with tab4:
    st.header("Modelo Bag of Words (Bolsa de Palabras)")
    corpus_input = st.text_area(
        "Ingresa frases separadas por saltos de línea para el corpus:",
        "El modelo de lenguaje aprende rápido.\nEl modelo procesa texto y tokens.\nGroq ofrece inferencia ultrarrápida de IA.",
        key="corpus_bow"
    )
    
    if corpus_input:
        documents = [doc.strip() for doc in corpus_input.split("\n") if doc.strip()]
        if len(documents) > 0:
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(documents)
            bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
            bow_df.index = [f"Doc {i+1}" for i in range(len(documents))]
            
            st.dataframe(bow_df, use_container_width=True)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(bow_df, annot=True, cmap="Purples", fmt="d", ax=ax)
            st.pyplot(fig)

# ----------------------------------------------------
# TAB 5: MÉTRICAS DE SIMILITUD
# ----------------------------------------------------
with tab5:
    st.header("Métricas de Similitud de Textos")
    text_a = st.text_input("Texto A:", "La inteligencia artificial y el aprendizaje automático transforman el mundo.")
    text_b = st.text_input("Texto B:", "El machine learning y la IA están cambiando la industria tecnológica.")
    
    if st.button("Calcular Similitud"):
        tfidf_vec = TfidfVectorizer()
        tfidf_matrix = tfidf_vec.fit_transform([text_a, text_b])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        st.metric(label="Similitud de Coseno", value=f"{similarity:.4f}")
        if similarity > 0.7:
            st.success("Los textos son altamente similares.")
        elif similarity > 0.3:
            st.warning("Los textos comparten algunos conceptos comunes.")
        else:
            st.info("Los textos son conceptualmente muy diferentes.")

# ----------------------------------------------------
# TAB 6: EMBEDDINGS
# ----------------------------------------------------
with tab6:
    st.header("Visualización de Embeddings Vectoriales")
    embedding_text = st.text_input("Frase o palabra para visualizar embeddings:", "Inteligencia Artificial")
    
    if st.button("Generar Vector de Embeddings"):
        np.random.seed(abs(hash(embedding_text)) % (2**32))
        vector_dim = 16
        simulated_embedding = np.random.uniform(-1.0, 1.0, vector_dim)
        
        df_emb = pd.DataFrame({
            "Dimensión": [f"Dim {i+1}" for i in range(vector_dim)],
            "Valor": simulated_embedding
        })
        
        st.dataframe(df_emb.T)
        
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(df_emb["Dimensión"], df_emb["Valor"], color="#7b1fa2")
        plt.xticks(rotation=45)
        ax.set_title(f"Espacio Vectorial para: '{embedding_text}'")
        st.pyplot(fig)
