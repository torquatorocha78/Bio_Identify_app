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
from typing import Dict, Optional

# =============================
# LOGGING
# =============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# CONFIG
# =============================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY não encontrada. Configure a variável de ambiente.")
    st.stop()

client = OpenAI(api_key=api_key)

st.set_page_config(
    page_title="Bio Identify - Seu Biólogo de Bolso",
    page_icon="🔬",
    layout="wide"
)

# =============================
# CSS
# =============================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

h1, h2, h3 {
    color: #667eea;
    font-weight: bold;
}

.info-box {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #1976d2;
}

.success-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #2e7d32;
}

.stats-metric {
    text-align: center;
    padding: 15px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =============================
# DATABASE MANAGER
# =============================

class DatabaseManager:
    """Gerenciador de banco de dados para identificações"""
    
    def __init__(self, db_path: str = "bio_identify.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Inicializa o banco de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS identifications(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        species_name TEXT NOT NULL,
                        common_name TEXT,
                        kingdom TEXT,
                        phylum TEXT,
                        class TEXT,
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
                
                cursor.execute("PRAGMA table_info(identifications)")
                columns = [col[1] for col in cursor.fetchall()]
                
                if "created_at" not in columns:
                    cursor.execute("""
                        ALTER TABLE identifications 
                        ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    """)
                    conn.commit()
                
                logger.info("Database initialized successfully")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def save_identification(self, data: Dict) -> bool:
        """Salva uma identificação no banco de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO identifications(
                        species_name, common_name, kingdom, phylum, class, 
                        order_name, family, genus, description, habitat, 
                        diet, conservation_status
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                logger.info(f"Identification saved: {data.get('species_name')}")
                return True
        except (sqlite3.Error, ValueError) as e:
            logger.error(f"Error saving identification: {e}")
            return False
    
    def get_history(self) -> pd.DataFrame:
        """Retorna o histórico de identificações"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(
                    """SELECT id, species_name, common_name, kingdom, class, 
                       COALESCE(created_at, datetime('now')) as created_at 
                       FROM identifications ORDER BY id DESC""",
                    conn
                )
                return df
        except sqlite3.Error as e:
            logger.error(f"Error loading history: {e}")
            return pd.DataFrame()
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas de identificações"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM identifications")
                total = cursor.fetchone()[0] or 0
                
                cursor.execute("""
                    SELECT kingdom, COUNT(*) 
                    FROM identifications 
                    GROUP BY kingdom
                """)
                kingdoms = dict(cursor.fetchall())
                
                cursor.execute("""
                    SELECT class, COUNT(*) 
                    FROM identifications 
                    WHERE class IS NOT NULL
                    GROUP BY class
                    ORDER BY COUNT(*) DESC
                    LIMIT 5
                """)
                classes = dict(cursor.fetchall())
                
                return {
                    "total": total,
                    "kingdoms": kingdoms,
                    "classes": classes
                }
        except sqlite3.Error as e:
            logger.error(f"Error getting statistics: {e}")
            return {}


# =============================
# SPECIES IDENTIFIER
# =============================

class SpeciesIdentifier:
    """Identificador de espécies com IA"""
    
    MAX_RETRIES = 3
    MODEL = "gpt-4o-mini"
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    @staticmethod
    def encode_image(image: Image.Image) -> str:
        """Codifica imagem para base64"""
        try:
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            buffered.seek(0)
            return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    def identify_species(self, image: Image.Image) -> Optional[Dict]:
        """Identifica espécies em foto com retry automático"""
        base64_image = self.encode_image(image)
        
        prompt = """
Analise esta imagem e identifique o ser vivo (animal, planta, fungo, etc).

Forneça informações sobre:
1. Nome científico (nomenclatura binomial)
2. Nome comum (em português)
3. Reino
4. Filo
5. Classe
6. Ordem
7. Família
8. Gênero
9. Descrição breve do ser vivo
10. Habitat/Ambiente onde vive
11. Alimentação/Dieta
12. Status de conservação (se conhecido)

Responda APENAS com JSON válido, sem markdown:
{
    "species_name": "Nome científico",
    "common_name": "Nome comum",
    "kingdom": "Reino",
    "phylum": "Filo",
    "class": "Classe",
    "order": "Ordem",
    "family": "Família",
    "genus": "Gênero",
    "description": "Descrição breve",
    "habitat": "Habitat/Ambiente",
    "diet": "Alimentação",
    "conservation_status": "Status de conservação"
}
"""
        
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                content = response.choices[0].message.content.strip()
                
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "")
                elif content.startswith("```"):
                    content = content.replace("```", "")
                
                data = json.loads(content)
                
                if not self._validate_data(data):
                    raise ValueError("Dados inválidos")
                
                logger.info(f"Species identification: {data['species_name']}")
                return data
                
            except RateLimitError:
                if attempt < self.MAX_RETRIES - 1:
                    st.warning(f"⏳ Tentativa {attempt + 1}/{self.MAX_RETRIES}...")
                    continue
                else:
                    raise
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error attempt {attempt + 1}: {e}")
                if attempt == self.MAX_RETRIES - 1:
                    raise
                continue
        
        return None
    
    @staticmethod
    def _validate_data(data: Dict) -> bool:
        """Valida dados"""
        required = ["species_name", "common_name", "kingdom"]
        return all(field in data and data[field] for field in required)


# =============================
# VALIDAÇÃO DE IMAGEM
# =============================

def validate_image(image: Image.Image) -> bool:
    """Valida imagem"""
    try:
        if image.size[0] < 100 or image.size[1] < 100:
            st.error("❌ Imagem muito pequena (mín: 100x100px)")
            return False
        
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        size_mb = len(buffer.getvalue()) / (1024 * 1024)
        
        if size_mb > 10:
            st.error("❌ Imagem muito grande (máx: 10MB)")
            return False
        
        return True
    except Exception as e:
        st.error(f"❌ Erro ao validar: {str(e)[:50]}")
        return False


# =============================
# VISUALIZAÇÕES (CORRIGIDAS)
# =============================

@st.cache_resource
def get_figure_kingdom(stats: Dict):
    """Cria gráfico de reino com cache"""
    if not stats.get("kingdoms"):
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    kingdoms = stats["kingdoms"]
    colors = ["#667eea", "#764ba2", "#f093fb", "#4facfe"]
    ax.barh(list(kingdoms.keys()), list(kingdoms.values()), color=colors[:len(kingdoms)])
    ax.set_xlabel("Quantidade")
    ax.set_title("Distribuição por Reino", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


@st.cache_resource
def get_figure_class(stats: Dict):
    """Cria gráfico de classe com cache"""
    if not stats.get("classes"):
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = stats["classes"]
    colors = ["#667eea", "#764ba2", "#f093fb", "#4facfe"]
    ax.pie(
        classes.values(),
        labels=classes.keys(),
        autopct="%1.1f%%",
        colors=colors[:len(classes)],
        startangle=90
    )
    ax.set_title("Top 5 Classes", fontweight="bold")
    plt.tight_layout()
    return fig


# =============================
# MAIN APP
# =============================

def main():
    # Inicializar sessão
    if "db" not in st.session_state:
        st.session_state.db = DatabaseManager()
    
    if "identifier" not in st.session_state:
        st.session_state.identifier = SpeciesIdentifier(client)
    
    if "show_success" not in st.session_state:
        st.session_state.show_success = False
    
    db = st.session_state.db
    identifier = st.session_state.identifier
    
    # Header
    st.title("🔬 Bio Identify - Seu Biólogo de Bolso")
    
    st.markdown("""
    <div class='info-box'>
    🌍 <b>Bio Identify</b> é seu assistente de identificação biológica inteligente. 
    Envie uma foto de qualquer ser vivo e obtenha informações científicas completas!
    </div>
    """, unsafe_allow_html=True)
    
    # Abas
    tab1, tab2, tab3 = st.tabs(["🔍 Identificar", "📊 Histórico", "📈 Estatísticas"])
    
    # =============================
    # TAB 1: IDENTIFICAR
    # =============================
    
    with tab1:
        st.subheader("📸 Envie uma Foto para Identificação")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Tire ou envie uma foto clara de um ser vivo**")
            file = st.file_uploader(
                "Escolha uma imagem",
                type=["jpg", "png", "jpeg"],
                key="upload_species"
            )
            
            if file:
                try:
                    image = Image.open(file)
                    
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    
                    st.image(image, caption="Foto enviada", width=300)
                    
                    if not validate_image(image):
                        st.stop()
                    
                    if st.button("🔍 Identificar Espécie", key="identify_btn"):
                        with st.spinner("⏳ IA analisando imagem..."):
                            try:
                                result = identifier.identify_species(image)
                                
                                if result and db.save_identification(result):
                                    st.session_state.show_success = True
                                    st.session_state.last_result = result
                                else:
                                    st.error("❌ Erro na identificação. Tente novamente.")
                            
                            except RateLimitError:
                                st.warning("⚠️ Limite atingido. Aguarde...")
                            except Exception as e:
                                st.error(f"❌ Erro: {str(e)[:80]}")
                
                except Exception as e:
                    st.error(f"❌ Erro ao abrir: {str(e)[:50]}")
        
        # Última identificação
        with col2:
            if st.session_state.show_success and "last_result" in st.session_state:
                st.markdown("""
                <div class='success-box'>
                ✅ <b>Identificação concluída!</b>
                </div>
                """, unsafe_allow_html=True)
                
                result = st.session_state.last_result
                st.markdown(f"### {result.get('common_name', 'N/A')}")
                st.markdown(f"**Científico:** *{result.get('species_name', 'N/A')}*")
                st.markdown(f"**Reino:** {result.get('kingdom', 'N/A')}")
                st.markdown(f"**Classe:** {result.get('class', 'N/A')}")
                st.markdown(f"**Descrição:** {result.get('description', 'N/A')[:150]}...")
            else:
                df = db.get_history()
                if len(df) > 0:
                    latest = df.iloc[0]
                    st.markdown("### 🏆 Última Identificação")
                    st.markdown(f"**{latest['common_name']}**")
                    st.markdown(f"*{latest['species_name']}*")
                    st.markdown(f"**Reino:** {latest['kingdom']}")
                else:
                    st.info("ℹ️ Nenhuma identificação ainda")
    
    # =============================
    # TAB 2: HISTÓRICO
    # =============================
    
    with tab2:
        st.subheader("📋 Histórico de Identificações")
        
        df = db.get_history()
        
        if len(df) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_kingdom = st.multiselect(
                    "Filtrar por Reino",
                    options=sorted(df["kingdom"].unique())
                )
            
            with col2:
                filter_class = st.multiselect(
                    "Filtrar por Classe",
                    options=sorted(df["class"].dropna().unique())
                )
            
            with col3:
                sort_by = st.radio("Ordenar", ["Recentes", "Antigas", "A-Z"])
            
            # Aplicar filtros
            filtered_df = df.copy()
            
            if filter_kingdom:
                filtered_df = filtered_df[filtered_df["kingdom"].isin(filter_kingdom)]
            
            if filter_class:
                filtered_df = filtered_df[filtered_df["class"].isin(filter_class)]
            
            if sort_by == "Antigas":
                filtered_df = filtered_df.iloc[::-1]
            elif sort_by == "A-Z":
                filtered_df = filtered_df.sort_values("common_name")
            
            if len(filtered_df) > 0:
                display_df = filtered_df.copy()
                display_df["created_at"] = pd.to_datetime(display_df["created_at"]).dt.strftime("%d/%m/%Y %H:%M")
                
                st.dataframe(
                    display_df[["id", "common_name", "species_name", "kingdom", "class", "created_at"]],
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
            else:
                st.info("ℹ️ Sem resultados com os filtros")
        else:
            st.info("ℹ️ Nenhuma identificação. Comece na aba 'Identificar'")
    
    # =============================
    # TAB 3: ESTATÍSTICAS
    # =============================
    
    with tab3:
        st.subheader("📊 Estatísticas")
        
        stats = db.get_statistics()
        
        if stats.get("total", 0) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Total", stats['total'])
            
            with col2:
                st.metric("🌍 Reinos", len(stats.get("kingdoms", {})))
            
            with col3:
                st.metric("🏆 Classes", len(stats.get("classes", {})))
            
            st.divider()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("### Distribuição por Reino")
                fig1 = get_figure_kingdom(stats)
                if fig1:
                    st.pyplot(fig1, use_container_width=True)
            
            with col2:
                st.write("### Top 5 Classes")
                fig2 = get_figure_class(stats)
                if fig2:
                    st.pyplot(fig2, use_container_width=True)
            
            st.divider()
            st.write("### Resumo por Reino")
            
            for kingdom, count in sorted(stats.get("kingdoms", {}).items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{kingdom}:** {count} espécie(s)")
                st.progress(min(count / stats['total'], 1.0))
        
        else:
            st.info("ℹ️ Sem dados. Identifique espécies primeiro!")


if __name__ == "__main__":
    main()
