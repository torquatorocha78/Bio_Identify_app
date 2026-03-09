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
from collections import Counter

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
    color: white;
}

.stMainBlockContainer {
    background-color: rgba(255, 255, 255, 0.98);
    color: #333;
}

h1, h2, h3 {
    color: #667eea;
    font-weight: bold;
}

.result-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #2e7d32;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.error-box {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #c62828;
}

.success-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #2e7d32;
}

.info-box {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #1976d2;
}

.kingdom-badge {
    display: inline-block;
    background: #667eea;
    color: white;
    padding: 5px 12px;
    border-radius: 20px;
    margin: 5px 5px 5px 0;
    font-size: 12px;
    font-weight: bold;
}

.species-card {
    background: white;
    padding: 15px;
    border-radius: 10px;
    border: 2px solid #667eea;
    margin: 10px 0;
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
                
                # Migração se necessário
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
                
                # Total de identificações
                cursor.execute("SELECT COUNT(*) FROM identifications")
                total = cursor.fetchone()[0] or 0
                
                # Distribuição por reino
                cursor.execute("""
                    SELECT kingdom, COUNT(*) 
                    FROM identifications 
                    GROUP BY kingdom
                """)
                kingdoms = dict(cursor.fetchall())
                
                # Distribuição por classe
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
        """
        Identifica espécies em foto com retry automático
        """
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

Se não conseguir identificar com certeza, indique o máximo que puder.

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
                                {
                                    "type": "text",
                                    "text": prompt
                                },
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
                
                # Limpar possível markdown
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "")
                elif content.startswith("```"):
                    content = content.replace("```", "")
                
                data = json.loads(content)
                
                # Validar dados
                if not self._validate_data(data):
                    raise ValueError("Dados inválidos retornados pela IA")
                
                logger.info(f"Species identification successful: {data['species_name']}")
                return data
                
            except RateLimitError:
                if attempt < self.MAX_RETRIES - 1:
                    st.warning(f"⏳ Rate limit atingido. Tentativa {attempt + 1}/{self.MAX_RETRIES}...")
                    continue
                else:
                    logger.error("Rate limit exceeded after retries")
                    raise
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error (attempt {attempt + 1}): {e}")
                if attempt == self.MAX_RETRIES - 1:
                    raise
                continue
            except Exception as e:
                logger.error(f"Identification error (attempt {attempt + 1}): {e}")
                if attempt == self.MAX_RETRIES - 1:
                    raise
                continue
        
        return None
    
    @staticmethod
    def _validate_data(data: Dict) -> bool:
        """Valida dados retornados pela IA"""
        required_fields = ["species_name", "common_name", "kingdom"]
        
        if not all(field in data for field in required_fields):
            return False
        
        # Verificar se campos obrigatórios têm conteúdo
        if not data["species_name"] or not data["common_name"]:
            return False
        
        return True


# =============================
# VALIDAÇÃO DE IMAGEM
# =============================

def validate_image(image: Image.Image) -> bool:
    """Valida imagem antes de processar"""
    try:
        if image.size[0] < 100 or image.size[1] < 100:
            st.error("❌ Imagem muito pequena. Use uma imagem maior que 100x100px")
            return False
        
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        size_mb = len(buffer.getvalue()) / (1024 * 1024)
        
        if size_mb > 10:
            st.error("❌ Imagem muito grande. Use uma imagem menor que 10MB")
            return False
        
        return True
    except Exception as e:
        st.error(f"❌ Erro ao validar imagem: {e}")
        logger.error(f"Image validation error: {e}")
        return False


# =============================
# VISUALIZAÇÕES
# =============================

def plot_kingdom_distribution(stats: Dict) -> plt.Figure:
    """Cria gráfico de distribuição por reino"""
    if not stats.get("kingdoms"):
        return None
    
    kingdoms = stats["kingdoms"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#667eea", "#764ba2", "#f093fb", "#4facfe", "#00f2fe"]
    ax.barh(list(kingdoms.keys()), list(kingdoms.values()), color=colors[:len(kingdoms)])
    ax.set_xlabel("Quantidade")
    ax.set_title("Distribuição de Identificações por Reino", fontweight="bold", fontsize=14)
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_class_distribution(stats: Dict) -> plt.Figure:
    """Cria gráfico de distribuição por classe"""
    if not stats.get("classes"):
        return None
    
    classes = stats["classes"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#667eea", "#764ba2", "#f093fb", "#4facfe", "#00f2fe"]
    ax.pie(
        classes.values(),
        labels=classes.keys(),
        autopct="%1.1f%%",
        colors=colors[:len(classes)],
        startangle=90
    )
    ax.set_title("Top 5 Classes Identificadas", fontweight="bold", fontsize=14)
    
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
    
    db = st.session_state.db
    identifier = st.session_state.identifier
    
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("### 🔬")
    with col2:
        st.title("Bio Identify - Seu Biólogo de Bolso")
    
    st.markdown("""
    <div class='info-box'>
    🌍 <b>Bio Identify</b> é seu assistente de identificação biológica inteligente. 
    Envie uma foto de qualquer ser vivo (animal, planta, fungo, etc) e obtenha 
    informações científicas completas sobre a espécie. Mantemos um histórico de 
    todas as suas descobertas!
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
            st.write("Tire ou envie uma foto clara de um ser vivo")
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
                    
                    st.image(image, caption="Foto enviada", use_column_width=True)
                    
                    if not validate_image(image):
                        st.stop()
                    
                    if st.button("🔍 Identificar Espécie", key="identify_btn", use_container_width=True):
                        with st.spinner("⏳ IA analisando imagem..."):
                            try:
                                result = identifier.identify_species(image)
                                
                                if result:
                                    if db.save_identification(result):
                                        st.markdown("""
                                        <div class='success-box'>
                                        ✅ <b>Identificação concluída com sucesso!</b>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        st.rerun()
                                    else:
                                        st.error("❌ Erro ao salvar identificação.")
                                else:
                                    st.error("❌ Não foi possível identificar. Tente outra foto.")
                            
                            except RateLimitError:
                                st.warning("⚠️ Limite de requisições atingido. Aguarde alguns momentos.")
                            except json.JSONDecodeError:
                                st.error("❌ Erro ao processar resposta da IA.")
                            except Exception as e:
                                st.error(f"❌ Erro: {str(e)[:100]}")
                                logger.error(f"Error: {e}", exc_info=True)
                
                except Exception as e:
                    st.error(f"❌ Erro ao abrir imagem: {str(e)}")
        
        # Mostrar última identificação na coluna 2
        with col2:
            df = db.get_history()
            if len(df) > 0:
                latest = df.iloc[0]
                st.markdown("### 🏆 Última Identificação")
                
                with st.container(border=True):
                    st.markdown(f"### {latest['common_name']}")
                    st.markdown(f"**Nome Científico:** *{latest['species_name']}*")
                    st.markdown(f"**Reino:** {latest['kingdom']}")
                    st.markdown(f"**Classe:** {latest['class']}")
                    st.markdown(f"**Data:** {pd.to_datetime(latest['created_at']).strftime('%d/%m/%Y %H:%M')}")
            else:
                st.info("ℹ️ Nenhuma identificação ainda. Comece enviando uma foto!")
    
    # =============================
    # TAB 2: HISTÓRICO
    # =============================
    
    with tab2:
        st.subheader("📋 Histórico de Identificações")
        
        df = db.get_history()
        
        if len(df) > 0:
            # Filtros
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_kingdom = st.multiselect(
                    "Filtrar por Reino",
                    options=df["kingdom"].unique(),
                    default=None
                )
            
            with col2:
                filter_class = st.multiselect(
                    "Filtrar por Classe",
                    options=df["class"].dropna().unique(),
                    default=None
                )
            
            with col3:
                sort_by = st.selectbox(
                    "Ordenar por",
                    ["Mais Recentes", "Mais Antigas", "A-Z"]
                )
            
            # Aplicar filtros
            filtered_df = df.copy()
            
            if filter_kingdom:
                filtered_df = filtered_df[filtered_df["kingdom"].isin(filter_kingdom)]
            
            if filter_class:
                filtered_df = filtered_df[filtered_df["class"].isin(filter_class)]
            
            # Aplicar ordenação
            if sort_by == "Mais Antigas":
                filtered_df = filtered_df.iloc[::-1]
            elif sort_by == "A-Z":
                filtered_df = filtered_df.sort_values("common_name")
            
            # Exibir dados
            if len(filtered_df) > 0:
                # Formatar para exibição
                display_df = filtered_df.copy()
                display_df["created_at"] = pd.to_datetime(display_df["created_at"]).dt.strftime("%d/%m/%Y %H:%M")
                
                display_df = display_df.rename(columns={
                    "id": "ID",
                    "species_name": "Nome Científico",
                    "common_name": "Nome Comum",
                    "kingdom": "Reino",
                    "class": "Classe",
                    "created_at": "Data"
                })
                
                st.dataframe(
                    display_df[["ID", "Nome Comum", "Nome Científico", "Reino", "Classe", "Data"]],
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("ℹ️ Nenhuma identificação encontrada com os filtros selecionados.")
        else:
            st.info("ℹ️ Nenhuma identificação ainda. Vá para 'Identificar' e envie uma foto!")
    
    # =============================
    # TAB 3: ESTATÍSTICAS
    # =============================
    
    with tab3:
        st.subheader("📊 Estatísticas e Análises")
        
        stats = db.get_statistics()
        
        if stats.get("total", 0) > 0:
            # Métricas principais
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class='stats-metric'>
                📊 Total de Identificações<br>
                <h2>{stats['total']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                num_kingdoms = len(stats.get("kingdoms", {}))
                st.markdown(f"""
                <div class='stats-metric'>
                🌍 Reinos Identificados<br>
                <h2>{num_kingdoms}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                num_classes = len(stats.get("classes", {}))
                st.markdown(f"""
                <div class='stats-metric'>
                🏆 Top Classes<br>
                <h2>{num_classes}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Gráficos
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("### 📊 Distribuição por Reino")
                fig1 = plot_kingdom_distribution(stats)
                if fig1:
                    st.pyplot(fig1, use_container_width=True)
            
            with col2:
                st.write("### 🥧 Top 5 Classes")
                fig2 = plot_class_distribution(stats)
                if fig2:
                    st.pyplot(fig2, use_container_width=True)
            
            st.divider()
            
            # Detalhes
            st.write("### 📈 Detalhes por Reino")
            
            for kingdom, count in sorted(stats.get("kingdoms", {}).items(), key=lambda x: x[1], reverse=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{kingdom}**")
                with col2:
                    st.write(f"**{count}** espécie(s)")
                st.progress(count / stats['total'])
        
        else:
            st.info("ℹ️ Sem dados para análise. Identifique espécies primeiro!")


if __name__ == "__main__":
    main()