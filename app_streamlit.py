import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Configuração da página

st.set_page_config(
    page_title="SafeDrive Vision",
    page_icon="👷",
    layout="wide",
)

DB_PATH = Path("data/safedrive.db")
LOG_PATH = Path("data/logs/runtime_predictions.csv")


# Funções de leitura de dados

def load_events_from_db():
    # Lê os eventos salvos no SQLite e retorna um DataFrame.
    # Retorna DataFrame vazio se o banco não existir.

    if not DB_PATH.exists():
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT * FROM drowsiness_events ORDER BY timestamp DESC",
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_events_from_csv():
    # Lê o log CSV gerado pelo worker_camera.py.
    # Usado como fallback se o SQLite estiver vazio.

    if not LOG_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(LOG_PATH)
    except Exception:
        return pd.DataFrame()


# Header

st.title("👷 SafeDrive Vision")
st.markdown("**Sistema de detecção de sonolência para operadores de empilhadeira**")
st.divider()

# Carrega dados

df_db = load_events_from_db()
df_csv = load_events_from_csv()

# Usa SQLite se tiver dados, senão usa CSV
if not df_db.empty:
    df = df_db
    data_source = "SQLite"
elif not df_csv.empty:
    df = df_csv
    data_source = "CSV"
else:
    df = pd.DataFrame()
    data_source = None

# Sidebar — filtros

st.sidebar.header("Filtros")

if not df.empty and "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

if not df.empty and "operator_id" in df.columns:
    operadores = ["Todos"] + sorted(df["operator_id"].dropna().unique().tolist())
    operador_selecionado = st.sidebar.selectbox("Operador", operadores)

    if operador_selecionado != "Todos":
        df = df[df["operator_id"] == operador_selecionado]

status_col = "prediction" if "prediction" in df.columns else "status" if "status" in df.columns else None

if not df.empty and status_col:
    status_opcoes = ["Todos"] + sorted(df[status_col].dropna().unique().tolist())
    status_selecionado = st.sidebar.selectbox("Status", status_opcoes)

    if status_selecionado != "Todos":
        df = df[df[status_col] == status_selecionado]

st.sidebar.divider()
st.sidebar.caption(f"Fonte dos dados: {data_source or 'Nenhuma'}")
st.sidebar.caption(f"Total de eventos: {len(df)}")

if st.sidebar.button("🔄 Atualizar dados"):
    st.rerun()

# Métricas principais

if df.empty:
    st.warning("Nenhum dado encontrado. Rode o worker_camera.py para gerar eventos.")
    st.info("Terminal 1: `python worker_camera.py`")
    st.stop()

col1, col2, col3, col4 = st.columns(4)

total = len(df)

if status_col:
    sonolento = len(df[df[status_col].isin(["SONOLENTO"])])
    normal = len(df[df[status_col].isin(["NORMAL"])])
    taxa_alerta = round((sonolento / total) * 100, 1) if total > 0 else 0
else:
    sonolento = 0
    normal = total
    taxa_alerta = 0

col1.metric("Total de eventos", total)
col2.metric("Normal", normal, delta=None)
col3.metric("Sonolento", sonolento, delta=None)
col4.metric("Taxa de alerta", f"{taxa_alerta}%")

st.divider()

# Gráficos

col_left, col_right = st.columns(2)

# Gráfico de pizza — distribuição de status
with col_left:
    st.subheader("Distribuição de status")

    if status_col and not df.empty:
        contagem = df[status_col].value_counts().reset_index()
        contagem.columns = ["Status", "Quantidade"]

        color_map = {
            "NORMAL": "#00cc66",
            "SONOLENTO": "#ff3333",
            "CALIBRANDO": "#ffaa00",
            "ROSTO NAO DETECTADO": "#aaaaaa",
        }

        fig_pizza = px.pie(
            contagem,
            names="Status",
            values="Quantidade",
            color="Status",
            color_discrete_map=color_map,
        )

        st.plotly_chart(fig_pizza, use_container_width=True)

# Gráfico de linha — EAR ao longo do tempo
with col_right:
    st.subheader("EAR ao longo do tempo")

    ear_col = "ear_mean" if "ear_mean" in df.columns else "mean_ear" if "mean_ear" in df.columns else None

    if ear_col and "timestamp" in df.columns:
        df_sorted = df.sort_values("timestamp")

        fig_ear = px.line(
            df_sorted,
            x="timestamp",
            y=ear_col,
            color=status_col if status_col else None,
            labels={"timestamp": "Tempo", ear_col: "EAR médio"},
            color_discrete_map={
                "NORMAL": "#00cc66",
                "SONOLENTO": "#ff3333",
            },
        )

        # Linha de referência do limiar de olho fechado
        fig_ear.add_hline(
            y=0.20,
            line_dash="dash",
            line_color="orange",
            annotation_text="Limiar (0.20)",
        )

        st.plotly_chart(fig_ear, use_container_width=True)
    else:
        st.info("Dados de EAR não disponíveis.")

st.divider()

# Gráfico de linha — PERCLOS ao longo do tempo
st.subheader("PERCLOS ao longo do tempo")

if "perclos" in df.columns and "timestamp" in df.columns:
    df_sorted = df.sort_values("timestamp")

    fig_perclos = px.area(
        df_sorted,
        x="timestamp",
        y="perclos",
        labels={"timestamp": "Tempo", "perclos": "PERCLOS"},
        color_discrete_sequence=["#ff6600"],
    )

    fig_perclos.add_hline(
        y=0.30,
        line_dash="dash",
        line_color="red",
        annotation_text="Alerta (30%)",
    )

    st.plotly_chart(fig_perclos, use_container_width=True)

st.divider()

# Tabela de eventos recentes

st.subheader("Eventos recentes")

colunas_exibir = [col for col in [
    "timestamp", status_col, "confidence", "ear_mean", "mean_ear",
    "perclos", "mar_mean", "mean_mar", "operator_id", "source"
] if col and col in df.columns]

df_display = df[colunas_exibir].head(50).copy()

if "timestamp" in df_display.columns:
    df_display["timestamp"] = df_display["timestamp"].astype(str).str[:19]

st.dataframe(df_display, use_container_width=True)