import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from datetime import datetime
from typing import Tuple, Dict, Any, List
import hashlib

# ==============================
# 🧠 CONFIGURACIÓN INICIAL
# ==============================
st.set_page_config(
    page_title="🚀 bueno profe, ahi lo cambiamo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 🧪 FUNCIONES (TODO EN MEMORIA)
# ==============================

def file_hash(file) -> str:
    """Genera un hash único del archivo para caché."""
    file.seek(0)
    return hashlib.md5(file.read()).hexdigest()

@st.cache_data(show_spinner=False)
def load_file_cached(file_hash: str, file_content: bytes, ext: str) -> pd.DataFrame:
    """Carga archivo desde bytes con caché por hash."""
    from io import BytesIO
    if ext == '.csv':
        return pd.read_csv(BytesIO(file_content))
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(BytesIO(file_content))
    elif ext == '.json':
        return pd.json_normalize(json.loads(file_content))
    else:
        raise ValueError(f"Extensión no soportada: {ext}")

def load_file(file) -> pd.DataFrame:
    ext = os.path.splitext(file.name)[1].lower()
    file.seek(0)
    content = file.read()
    h = hashlib.md5(content).hexdigest()
    return load_file_cached(h, content, ext)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('-', '_')
        .str.replace(r'[^\w]', '', regex=True)
    )
    return df

def clean_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    df = standardize_columns(df)
    before_rows = len(df)
    df = df.drop_duplicates()
    
    # Llenar nulos en strings
    obj_cols = df.select_dtypes(include=['object', 'string']).columns
    for c in obj_cols:
        df[c] = df[c].fillna('Desconocido')
    
    # Intentar convertir columnas que parecen numéricas
    for col in obj_cols:
        if df[col].astype(str).str.contains(r'^[\d\.\,\-\s\$€£]+$', na=False).all():
            temp = df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
            numeric_series = pd.to_numeric(temp, errors='coerce')
            if numeric_series.notna().any():
                df[col + '_num'] = numeric_series
    
    after_rows = len(df)
    report = {
        'rows_before': before_rows,
        'rows_after': after_rows,
        'deduplicated': before_rows - after_rows,
        'null_filled_cols': list(obj_cols)
    }
    return df, report

def auto_detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Detecta columnas clave con lógica mejorada."""
    cols = df.columns.str.lower().tolist()
    mapping = {}
    
    # Monto: busca en orden de prioridad
    for col in cols:
        if any(k in col for k in ['venta', 'monto', 'total', 'ingreso', 'valor', 'amount', 'revenue', 'precio', 'costo']):
            mapping['monto'] = col
            break
    else:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        mapping['monto'] = numeric_cols[0] if numeric_cols else None

    # Fecha
    for col in cols:
        if any(k in col for k in ['fecha', 'date', 'dia', 'día', 'time', 'timestamp']):
            mapping['fecha'] = col
            break
    else:
        # ¿Alguna columna se ve como fecha?
        for col in df.select_dtypes(include='object').columns:
            sample = df[col].dropna().head(3)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample, errors='raise')
                    mapping['fecha'] = col
                    break
                except:
                    continue
        else:
            mapping['fecha'] = None

    # Dimensiones
    keywords = {
        'producto': ['producto', 'item', 'articulo', 'sku', 'nombre', 'servicio', 'titulo', 'descripcion'],
        'local': ['local', 'tienda', 'sucursal', 'store', 'branch', 'oficina'],
        'region': ['region', 'ciudad', 'zona', 'pais', 'departamento', 'estado', 'provincia'],
        'cliente': ['cliente', 'customer', 'nombre_cliente', 'id_cliente', 'usuario', 'user']
    }

    for key, words in keywords.items():
        for col in cols:
            if any(w in col for w in words):
                mapping[key] = col
                break
        else:
            mapping[key] = None

    return mapping

# ==============================
# 🚀 INTERFAZ DE USUARIO
# ==============================

st.title("⚡ Dashboard Mutado Universal — Análisis Inteligente sin Límites")
st.markdown("Sube archivos **CSV, Excel o JSON** y obtén insights automáticos. ¡Sin etiquetas forzadas!")

with st.sidebar:
    st.header("📁 Carga de Archivos")
    uploaded_files = st.file_uploader(
        "Sube tus archivos", 
        type=['csv','xlsx','xls','json'], 
        accept_multiple_files=True,
        help="Soporta múltiples archivos. Funciona con cualquier tipo de dato."
    )
    st.divider()
    st.caption("💡 Tip: Usa nombres claros como 'datos_2024.csv'")

if not uploaded_files:
    st.info("👆 Sube al menos un archivo para comenzar.")
    st.stop()

# ==============================
# 🧹 PROCESAMIENTO UNIVERSAL (SIN ÁREAS)
# ==============================

all_clean_dfs = []

for file in uploaded_files:
    with st.expander(f"📂 {file.name}", expanded=False):
        try:
            df = load_file(file)
            st.write(f"✅ Cargado: `{df.shape[0]}` filas, `{df.shape[1]}` columnas")
            df_clean, report = clean_table(df)
            st.json(report)
            all_clean_dfs.append(df_clean)
        except Exception as e:
            st.error(f"❌ Error en {file.name}: {str(e)}")
            continue

# Combinar todos los archivos
if not all_clean_dfs:
    st.error("⚠️ No se pudieron cargar archivos válidos.")
    st.stop()

df_master = pd.concat(all_clean_dfs, ignore_index=True)

# Intentar convertir columnas object a datetime si se parecen
obj_cols = df_master.select_dtypes(include='object').columns
for col in obj_cols:
    sample = df_master[col].dropna().head(5)
    if len(sample) > 0:
        try:
            pd.to_datetime(sample, errors='raise')
            df_master[col] = pd.to_datetime(df_master[col], errors='coerce')
        except:
            pass

# Validar que haya al menos algo útil
numeric_cols = df_master.select_dtypes(include='number').columns.tolist()
datetime_cols = df_master.select_dtypes(include='datetime').columns.tolist()

if not numeric_cols and not datetime_cols:
    st.warning("⚠️ No se detectaron columnas numéricas ni de fecha. Intenta con otro archivo.")
    st.dataframe(df_master.head(), use_container_width=True)
    st.stop()

# ==============================
# 🔍 DETECCIÓN INTELIGENTE DE COLUMNAS
# ==============================

col_map = auto_detect_columns(df_master)

# Si no hay monto detectado pero hay numéricas, usar la primera
if not col_map.get('monto') and numeric_cols:
    col_map['monto'] = numeric_cols[0]

# Si no hay fecha detectada pero hay datetime, usar la primera
if not col_map.get('fecha') and datetime_cols:
    col_map['fecha'] = datetime_cols[0]

# ==============================
# 🎛️ FILTROS PRINCIPALES (EN SIDEBAR)
# ==============================

st.sidebar.header("🎛️ Filtros Globales")

# --- Conversión de monto a numérico si es necesario ---
if col_map['monto'] and df_master[col_map['monto']].dtype == 'object':
    cleaned = df_master[col_map['monto']].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
    df_master[col_map['monto']] = pd.to_numeric(cleaned, errors='coerce')
    df_master = df_master.dropna(subset=[col_map['monto']])

# --- Conversión y manejo de fechas ---
if col_map['fecha'] and col_map['fecha'] in df_master.columns:
    if df_master[col_map['fecha']].dtype == 'object':
        df_master[col_map['fecha']] = pd.to_datetime(df_master[col_map['fecha']], errors='coerce')
    df_master = df_master.dropna(subset=[col_map['fecha']])
    df_master['año'] = df_master[col_map['fecha']].dt.year
    df_master['mes'] = df_master[col_map['fecha']].dt.to_period('M').astype(str)

    # Filtro por año
    años = sorted(df_master['año'].dropna().unique())
    if len(años) > 1:
        año_sel = st.sidebar.radio("Año", ["Todos"] + [int(y) for y in años], horizontal=True)
        if año_sel != "Todos":
            df_master = df_master[df_master['año'] == int(año_sel)]
    
    # Filtro por mes
    if col_map['fecha'] in df_master.columns and len(df_master) > 0:
        meses = sorted(df_master['mes'].dropna().unique())
        if len(meses) > 1:
            mes_sel = st.sidebar.multiselect("Mes", meses)
            if mes_sel:
                df_master = df_master[df_master['mes'].isin(mes_sel)]

# --- Filtros por dimensiones ---
for dim_name, col in [('Producto', col_map['producto']), ('Local', col_map['local']), ('Región', col_map['region']), ('Cliente', col_map['cliente'])]:
    if col and col in df_master.columns:
        unique_vals = sorted(df_master[col].dropna().astype(str).unique())
        if 0 < len(unique_vals) <= 100:
            selected = st.sidebar.multiselect(f"Filtrar por {dim_name}", unique_vals)
            if selected:
                df_master = df_master[df_master[col].astype(str).isin(selected)]

# ==============================
# 📊 KPIs DINÁMICOS
# ==============================

if col_map['monto'] and col_map['monto'] in df_master.columns:
    total = df_master[col_map['monto']].sum()
    avg = df_master[col_map['monto']].mean()
    count = len(df_master)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total", f"${total:,.0f}")
    col2.metric("🧾 Promedio", f"${avg:,.0f}")
    col3.metric("🧮 Registros", f"{count:,}")

# ==============================
# 📈 ANÁLISIS AUTOMÁTICO
# ==============================

st.subheader("🤖 Análisis Inteligente")

# Tendencia temporal
if col_map['fecha'] and col_map['monto'] and col_map['fecha'] in df_master.columns and col_map['monto'] in df_master.columns:
    try:
        df_trend = df_master.set_index(col_map['fecha']).resample('M')[col_map['monto']].sum().reset_index()
        if len(df_trend) > 1:
            fig = px.line(df_trend, x=col_map['fecha'], y=col_map['monto'], title="📈 Tendencia Temporal")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudo generar la tendencia: {e}")

# Top N por dimensiones
dimensions = {
    "Producto": col_map['producto'],
    "Local": col_map['local'],
    "Región": col_map['region'],
    "Cliente": col_map['cliente']
}

for name, col in dimensions.items():
    if col and col in df_master.columns and col_map['monto'] and col_map['monto'] in df_master.columns:
        st.markdown(f"#### 🥇 Top 10 por {name}")
        df_top = df_master.groupby(col)[col_map['monto']].sum().reset_index().sort_values(col_map['monto'], ascending=False).head(10)
        fig = px.bar(df_top, x=col, y=col_map['monto'], title=f"Top 10 {name}", color=col_map['monto'])
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# 🎨 GRÁFICOS PERSONALIZADOS
# ==============================

st.divider()
st.subheader("🎨 Crea Tu Propio Gráfico")

numeric_cols = df_master.select_dtypes(include='number').columns.tolist()
all_cols = df_master.columns.tolist()

if numeric_cols and all_cols:
    col_x = st.selectbox("Eje X", all_cols, key="custom_x")
    col_y = st.selectbox("Eje Y", numeric_cols, key="custom_y")
    chart_type = st.radio("Tipo", ["Barras", "Líneas", "Pastel", "Dispersión"], horizontal=True, key="chart_type")

    if col_x and col_y:
        try:
            if chart_type == "Barras":
                df_agg = df_master.groupby(col_x)[col_y].sum().reset_index()
                fig = px.bar(df_agg, x=col_x, y=col_y, title=f"{col_y} por {col_x}")
            elif chart_type == "Líneas":
                if col_map['fecha'] and col_x == col_map['fecha']:
                    df_agg = df_master.set_index(col_x).resample('M')[col_y].sum().reset_index()
                else:
                    df_agg = df_master.groupby(col_x)[col_y].sum().reset_index()
                fig = px.line(df_agg, x=col_x, y=col_y, title=f"{col_y} por {col_x}")
            elif chart_type == "Pastel":
                df_agg = df_master.groupby(col_x)[col_y].sum().reset_index()
                fig = px.pie(df_agg, names=col_x, values=col_y, title=f"Distribución de {col_y} por {col_x}")
            elif chart_type == "Dispersión":
                fig = px.scatter(df_master, x=col_x, y=col_y, opacity=0.7, title=f"{col_y} vs {col_x}")
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error al generar gráfico: {e}")

# ==============================
# 📥 DESCARGA DE DATOS
# ==============================

st.divider()
st.subheader("📥 Exportar Resultados")

st.download_button(
    label="💾 Descargar Datos Filtrados (CSV)",
    data=df_master.to_csv(index=False).encode('utf-8'),
    file_name=f"dashboard_filtrado_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)

# ==============================
# ℹ️ INFORMACIÓN TÉCNICA
# ==============================

with st.expander("🔍 Metadatos y Diagnóstico"):
    st.write("### Columnas detectadas automáticamente:")
    st.json(col_map)
    st.write("### Tipos de datos:")
    st.write(df_master.dtypes.to_dict())
    st.write("### Vista previa de datos finales:")

    st.dataframe(df_master.head(10), use_container_width=True)
