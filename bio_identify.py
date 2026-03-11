import streamlit as st
from openai import OpenAI
from PIL import Image
import base64
import io
import os
import sqlite3
import pandas as pd
import json
from dotenv import load_dotenv

# =============================
# CONFIG
# =============================
load_dotenv()

# Validar API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY não encontrada. Configure a variável de ambiente.")
    st.stop()

client = OpenAI(api_key=api_key)

# =============================
# DATABASE
# =============================

def init_db():
    conn = sqlite3.connect("bio_identify.db")
    conn.execute("""
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
    conn = sqlite3.connect("bio_identify.db")
    conn.execute("""
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

def get_history():
    try:
        conn = sqlite3.connect("bio_identify.db")
        df = pd.read_sql("SELECT * FROM identifications ORDER BY id DESC LIMIT 100", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

# =============================
# MAIN
# =============================

init_db()

st.title("🔬 Identificação Biológica")
st.write("Seu Biólogo de Bolso")
st.write("Identifique seres vivos com IA")

# =============================
# IDENTIFY TAB
# =============================

st.subheader("📸 Envie uma Foto")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        st.image(image, caption="Foto", use_column_width=True)
        
        if st.button("Identificar"):
            try:
                with st.spinner("Analisando..."):
                    # Encode image
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG", quality=80)
                    base64_image = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Call API
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Identifique o ser vivo. JSON: {\"species_name\": \"\", \"common_name\": \"\", \"kingdom\": \"\", \"phylum\": \"\", \"class\": \"\", \"order\": \"\", \"family\": \"\", \"genus\": \"\", \"description\": \"\", \"habitat\": \"\", \"diet\": \"\", \"conservation_status\": \"\"}"},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }],
                        temperature=0.3,
                        max_tokens=400
                    )
                    
                    content = response.choices[0].message.content.strip()
                    if "```" in content:
                        content = content.replace("```json", "").replace("```", "").strip()
                    
                    result = json.loads(content)
                    save_identification(result)
                    
                    st.success("✅ Salvo!")
                    st.write(f"**{result.get('common_name', 'N/A')}**")
                    st.write(f"*{result.get('species_name', 'N/A')}*")
                    st.write(f"**Reino:** {result.get('kingdom', 'N/A')}")
                    st.write(f"**Descrição:** {result.get('description', 'N/A')}")
                    
            except Exception as e:
                st.error(f"Erro: {str(e)}")

with col2:
    st.write("### Dicas")
    st.info("✅ Foto clara\n✅ Boa luz\n✅ Mín 100x100px\n✅ Máx 10MB")

# =============================
# HISTORY TAB
# =============================

st.divider()
st.subheader("📋 Histórico")

df = get_history()

if len(df) > 0:
    col1, col2 = st.columns(2)
    
    with col1:
        filter_kingdom = st.selectbox("Filtrar Reino", ["Todos"] + list(df["kingdom"].unique()))
    
    with col2:
        filter_class = st.selectbox("Filtrar Classe", ["Todos"] + list(df["class_name"].dropna().unique()))
    
    filtered = df.copy()
    
    if filter_kingdom != "Todos":
        filtered = filtered[filtered["kingdom"] == filter_kingdom]
    
    if filter_class != "Todos":
        filtered = filtered[filtered["class_name"] == filter_class]
    
    if len(filtered) > 0:
        display = filtered[["id", "common_name", "species_name", "kingdom", "class_name"]].copy()
        display.columns = ["ID", "Nome Comum", "Nome Científico", "Reino", "Classe"]
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("Sem resultados")
else:
    st.info("Sem identificações")
