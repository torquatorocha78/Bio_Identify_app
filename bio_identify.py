import streamlit as st
from openai import OpenAI, RateLimitError
from PIL import Image
import base64
import io
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import json
import logging

# =============================
# CONFIG
# =============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

api_key = os.getenv("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")
if not api_key:
    st.error("❌ OPENAI_API_KEY não configurada!")
    st.stop()

client = OpenAI(api_key=api_key)

st.set_page_config(
    page_title="Bio Identify",
    page_icon="🔬",
    layout="wide"
)

# =============================
# CSS SIMPLES
# =============================

st.markdown("""
<style>
body { background-color: #f0f2f6; }
h1 { color: #667eea; }
</style>
""", unsafe_allow_html=True)

# =============================
# DATABASE
# =============================

def init_db():
    conn = sqlite3.connect("bio_identify.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS identifications(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            species_name TEXT,
            common_name TEXT,
            kingdom TEXT,
            phylum TEXT,
            class_name TEXT,
            order_name TEXT,
            family TEXT,
            genus TEXT,
            description TEXT,
            habitat TEXT,
            diet TEXT,
            conservation_status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_identification(data):
    try:
        conn = sqlite3.connect("bio_identify.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO identifications(
                species_name, common_name, kingdom, phylum, class_name,
                order_name, family, genus, description, habitat, diet, conservation_status
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("species_name", ""),
            data.get("common_name", ""),
            data.get("kingdom", ""),
            data.get("phylum", ""),
            data.get("class", ""),
            data.get("order", ""),
            data.get("family", ""),
            data.get("genus", ""),
            data.get("description", ""),
            data.get("habitat", ""),
            data.get("diet", ""),
            data.get("conservation_status", "")
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error saving: {e}")
        return False

def get_history():
    try:
        conn = sqlite3.connect("bio_identify.db")
        df = pd.read_sql("SELECT * FROM identifications ORDER BY id DESC", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

def get_stats():
    try:
        conn = sqlite3.connect("bio_identify.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM identifications")
        total = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT kingdom, COUNT(*) FROM identifications GROUP BY kingdom")
        kingdoms = dict(cursor.fetchall())
        
        cursor.execute("SELECT class_name, COUNT(*) FROM identifications WHERE class_name IS NOT NULL GROUP BY class_name ORDER BY COUNT(*) DESC LIMIT 5")
        classes = dict(cursor.fetchall())
        
        conn.close()
        return {"total": total, "kingdoms": kingdoms, "classes": classes}
    except:
        return {}

# =============================
# AI IDENTIFIER
# =============================

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=80)
    return base64.b64encode(buffered.getvalue()).decode()

def identify_species(image):
    base64_image = encode_image(image)
    
    prompt = """Analise esta imagem e identifique o ser vivo. Retorne APENAS JSON:
{
    "species_name": "Nome científico",
    "common_name": "Nome comum em português",
    "kingdom": "Reino",
    "phylum": "Filo",
    "class": "Classe",
    "order": "Ordem",
    "family": "Família",
    "genus": "Gênero",
    "description": "Descrição breve",
    "habitat": "Habitat",
    "diet": "Dieta",
    "conservation_status": "Status"
}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            temperature=0.3,
            max_tokens=400
        )
        
        content = response.choices[0].message.content.strip()
        
        # Limpar markdown
        if "```" in content:
            content = content.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(content)
        return data
    except Exception as e:
        logger.error(f"Error: {e}")
        return None

# =============================
# MAIN APP
# =============================

init_db()

st.title("🔬 Bio Identify - Seu Biólogo de Bolso")
st.write("Identifique qualquer ser vivo com IA!")

# Menu
menu = st.radio("Escolha uma opção:", ["Identificar", "Histórico", "Estatísticas"], horizontal=True)

# =============================
# IDENTIFICAR
# =============================

if menu == "Identificar":
    st.subheader("📸 Envie uma Foto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            st.image(image, caption="Foto", width=300)
            
            # Validar
            if image.size[0] < 100 or image.size[1] < 100:
                st.error("❌ Imagem muito pequena")
            else:
                if st.button("🔍 Identificar"):
                    with st.spinner("Analisando..."):
                        result = identify_species(image)
                        
                        if result:
                            if save_identification(result):
                                st.success("✅ Salvo com sucesso!")
                                
                                st.write(f"### {result.get('common_name', 'N/A')}")
                                st.write(f"**Científico:** {result.get('species_name', 'N/A')}")
                                st.write(f"**Reino:** {result.get('kingdom', 'N/A')}")
                                st.write(f"**Classe:** {result.get('class', 'N/A')}")
                                st.write(f"**Descrição:** {result.get('description', 'N/A')}")
                                st.write(f"**Habitat:** {result.get('habitat', 'N/A')}")
                                st.write(f"**Dieta:** {result.get('diet', 'N/A')}")
                        else:
                            st.error("❌ Erro na identificação")
    
    with col2:
        st.write("### 📌 Dicas")
        st.info("""
        ✅ Use fotos claras
        ✅ Boa iluminação
        ✅ Foco no ser vivo
        ✅ Mín 100x100px
        ✅ Máx 10MB
        """)

# =============================
# HISTÓRICO
# =============================

elif menu == "Histórico":
    st.subheader("📋 Histórico de Identificações")
    
    df = get_history()
    
    if len(df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            filter_kingdom = st.multiselect("Filtrar Reino", df["kingdom"].unique())
        
        with col2:
            filter_class = st.multiselect("Filtrar Classe", df["class_name"].dropna().unique())
        
        filtered = df.copy()
        
        if filter_kingdom:
            filtered = filtered[filtered["kingdom"].isin(filter_kingdom)]
        
        if filter_class:
            filtered = filtered[filtered["class_name"].isin(filter_class)]
        
        if len(filtered) > 0:
            display = filtered[["id", "common_name", "species_name", "kingdom", "class_name", "created_at"]].copy()
            display.columns = ["ID", "Nome Comum", "Nome Científico", "Reino", "Classe", "Data"]
            
            st.dataframe(display, use_container_width=True)
        else:
            st.info("Sem resultados")
    else:
        st.info("Nenhuma identificação ainda")

# =============================
# ESTATÍSTICAS
# =============================

elif menu == "Estatísticas":
    st.subheader("📊 Estatísticas")
    
    stats = get_stats()
    
    if stats.get("total", 0) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total", stats["total"])
        
        with col2:
            st.metric("Reinos", len(stats.get("kingdoms", {})))
        
        with col3:
            st.metric("Classes", len(stats.get("classes", {})))
        
        st.divider()
        
        # Gráfico Reino
        if stats.get("kingdoms"):
            fig, ax = plt.subplots(figsize=(10, 5))
            kingdoms = stats["kingdoms"]
            ax.barh(list(kingdoms.keys()), list(kingdoms.values()), color="#667eea")
            ax.set_xlabel("Quantidade")
            ax.set_title("Por Reino")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        st.divider()
        
        # Gráfico Classes
        if stats.get("classes"):
            fig, ax = plt.subplots(figsize=(10, 5))
            classes = stats["classes"]
            ax.pie(classes.values(), labels=classes.keys(), autopct="%1.1f%%")
            ax.set_title("Top 5 Classes")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    
    else:
        st.info("Sem dados")
