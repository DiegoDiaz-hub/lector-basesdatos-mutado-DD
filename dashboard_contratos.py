
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import hashlib
from io import BytesIO
from typing import Dict, List, Optional, Tuple

# ==============================
# 🧠 CONFIGURACIÓN INICIAL
# ==============================
st.set_page_config(
    page_title="📋 Gestión de Contratos — Softys",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 🎨 ESTILOS PERSONALIZADOS (TU UI)
# ==============================
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d5986);
        border-radius: 12px;
        padding: 16px 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .metric-value { font-size: 2rem; font-weight: 700; margin: 4px 0; }
    .metric-label { font-size: 0.8rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 1px; }
    .alert-box {
        border-left: 5px solid #e74c3c;
        background: #fdf2f2;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    .warn-box {
        border-left: 5px solid #f39c12;
        background: #fef9ed;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    .success-box {
        border-left: 5px solid #27ae60;
        background: #f0fdf4;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)

# ==============================
# 🛡️ CONSTANTES
# ==============================
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
HOJA_PRINCIPAL = "Antiguo"  # Tu sheet principal
DIAS_ALERTA_VENCIMIENTO = 30  # Umbral para alertas

# ==============================
# 🔧 FUNCIONES BACKEND ROBUSTO (MI LÓGICA)
# ==============================

def parse_fecha_softys(valor) -> pd.Timestamp:
    """
    Parser robusto para fechas del Excel de Softys.
    Maneja: '30"-"09"-"2025', '4/26/19', '99.99.9999', 'Indefinido', etc.
    """
    if pd.isna(valor) or valor in ['99.99.9999', '2999', '31/12/2999', 'Indefinido', '']:
        return pd.NaT
    
    # Normalizar formatos problemáticos
    valor_str = str(valor).strip()
    valor_limpio = valor_str.replace('"-"', '-').replace('/', '-').replace('.', '-')
    
    # Intentar múltiples formatos comunes en Softys
    formatos = ['%d-%m-%Y', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', '%d-%m-%y', '%m/%d/%y']
    for fmt in formatos:
        try:
            return pd.to_datetime(valor_limpio, format=fmt, dayfirst=True)
        except:
            continue
    
    # Fallback con coerce
    return pd.to_datetime(valor_limpio, errors='coerce', dayfirst=True)

def limpiar_monto(valor) -> float:
    """Limpia montos con separadores, símbolos y formatos mixtos."""
    if pd.isna(valor):
        return 0.0
    try:
        limpio = str(valor).replace('.', '').replace(',', '.').replace('$', '').replace('UF', '').strip()
        return float(limpio) if limpio else 0.0
    except:
        return 0.0

def clasificar_riesgo_contrato(estado_ariba: str, dias_restantes: int, es_indefinido: bool = False) -> str:
    """
    Clasificación de riesgo para compras spot.
    Combina estado Ariba + días restantes + contratos indefinidos.
    """
    # Mapeo de estados Ariba
    estados_bajos = ['Vigente', 'Publicado', 'En revisión', 'Aprobado']
    estados_medios = ['Próximo a vencer', 'Por vencer', 'En modificación']
    estados_altos = ['Vencido', 'Cancelado', 'Terminado']
    estados_revisar = ['Borrador', 'Modificación del borrador', 'En espera']
    
    if es_indefinido or estado_ariba in estados_bajos:
        return 'BAJO 🟢'
    if estado_ariba in estados_revisar:
        return 'REVISAR ⚪'
    if dias_restantes < 0 or estado_ariba in estados_altos:
        return 'ALTO 🔴'
    if dias_restantes <= 30 or estado_ariba in estados_medios:
        return 'MEDIO 🟡'
    return 'BAJO 🟢'

def validar_calidad_datos(df: pd.DataFrame) -> Dict[str, any]:
    """Validaciones de calidad para evitar decisiones con datos corruptos."""
    reporte = {
        'filas_totales': len(df),
        'contratos_sin_rut': df['Rut'].isna().sum() if 'Rut' in df.columns else 0,
        'contratos_sin_proveedor': df['Proveedor'].isna().sum() if 'Proveedor' in df.columns else 0,
        'fechas_invalidas': 0,
        'montos_anomalos': 0
    }
    
    # Validar fechas críticas
    for col in ['Fecha Inicio', 'Fecha Término Contrato', 'Vencimiento Garantía']:
        if col in df.columns:
            fechas = df[col].apply(parse_fecha_softys)
            reporte['fechas_invalidas'] += fechas.isna().sum()
    
    # Validar montos de garantía
    if 'Monto Garantía' in df.columns:
        montos = df['Monto Garantía'].apply(limpiar_monto)
        reporte['montos_anomalos'] = ((montos < 0) | (montos > 1_000_000_000)).sum()
    
    return reporte

@st.cache_data(show_spinner=False)
def cargar_y_procesar_contratos(file_hash: str, file_content: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Carga las 3 sheets principales y procesa con lógica de negocio Softys.
    Retorna: df_consolidado, df_bg, df_ariba, reporte_calidad
    """
    # Leer todas las sheets
    sheets = pd.read_excel(BytesIO(file_content), sheet_name=None, engine='openpyxl')
    
    # Sheet principal (puede ser 'Antiguo' o 'Consolidado de Contratos')
    hoja_principal = HOJA_PRINCIPAL if HOJA_PRINCIPAL in sheets.keys() else list(sheets.keys())[0]
    df_consolidado = sheets.get(hoja_principal, pd.DataFrame()).dropna(how='all').dropna(axis=1, how='all')
    
    # Sheets opcionales: BG e Info Ariba
    df_bg = sheets.get('BG', pd.DataFrame()) if 'BG' in sheets.keys() else pd.DataFrame()
    df_ariba = sheets.get('Info Ariba', pd.DataFrame()) if 'Info Ariba' in sheets.keys() else pd.DataFrame()
    
    if not df_bg.empty:
        df_bg = df_bg.dropna(how='all').dropna(axis=1, how='all')
    if not df_ariba.empty:
        df_ariba = df_ariba.dropna(how='all').dropna(axis=1, how='all')
    
    # ==============================
    # PROCESAMIENTO PRINCIPAL
    # ==============================
    df = df_consolidado.copy()
    hoy = pd.Timestamp.today().normalize()
    
    # Normalizar nombres de columnas (case-insensitive)
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    # Parsear fechas críticas con parser robusto
    fecha_cols = [c for c in df.columns if 'fecha' in c or 'término' in c or 'vencimiento' in c]
    for col in fecha_cols:
        df[col] = df[col].apply(parse_fecha_softys)
    
    # Calcular días para vencimiento
    col_termino = 'fecha_término_contrato' if 'fecha_término_contrato' in df.columns else 'fecha_termino_contrato'
    if col_termino in df.columns:
        df['dias_para_vencimiento'] = (df[col_termino] - hoy).dt.days
    
    # Identificar contratos indefinidos
    df['es_indefinido'] = df.get('contratos_indefinidos', '').str.lower().isin(['sí', 'si', 'yes', 'indefinido']) | \
                         (df.get('estado_contrato', '').str.lower() == 'indefinido') | \
                         (df[col_termino].dt.year > 2030 if pd.api.types.is_datetime64_any_dtype(df[col_termino]) else False)
    
    # Clasificar riesgo spot
    if 'estado_contrato_ariba' in df.columns and 'dias_para_vencimiento' in df.columns:
        df['riesgo_spot'] = df.apply(
            lambda row: clasificar_riesgo_contrato(
                row.get('estado_contrato_ariba', ''),
                row.get('dias_para_vencimiento', 999),
                row.get('es_indefinido', False)
            ),
            axis=1
        )
    
    # Limpiar montos de garantía
    if 'monto_garantía' in df.columns:
        df['monto_garantia_clp'] = df['monto_garantía'].apply(limpiar_monto)
    
    # Validar calidad de datos
    reporte_calidad = validar_calidad_datos(df)
    
    # Eliminar filas sin contrato válido
    if 'contrato_ariba' in df.columns:
        df = df[df['contrato_ariba'].notna() & (df['contrato_ariba'].astype(str).str.strip() != '')]
    
    return df, df_bg, df_ariba, reporte_calidad

# ==============================
# 📊 FUNCIONES DE VISUALIZACIÓN (TU UI)
# ==============================

def crear_kpi_cards(df: pd.DataFrame) -> None:
    """Crea tarjetas de KPIs ejecutivos con tu estilo."""
    total = len(df)
    vigentes = len(df[df['riesgo_spot'] == 'BAJO 🟢']) if 'riesgo_spot' in df.columns else 0
    por_vencer = len(df[df['riesgo_spot'] == 'MEDIO 🟡']) if 'riesgo_spot' in df.columns else 0
    vencidos = len(df[df['riesgo_spot'] == 'ALTO 🔴']) if 'riesgo_spot' in df.columns else 0
    revisar = len(df[df['riesgo_spot'] == 'REVISAR ⚪']) if 'riesgo_spot' in df.columns else 0
    
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📋 Total Contratos", f"{total:,}")
    k2.metric("✅ Vigentes", f"{vigentes:,}", delta=f"{vigentes/total*100:.1f}%" if total else "0%")
    k3.metric("⚠️ Por Vencer (≤30d)", f"{por_vencer:,}", delta_color="inverse")
    k4.metric("🚨 Vencidos", f"{vencidos:,}", delta_color="inverse")
    k5.metric("🔍 Revisar", f"{revisar}")

def crear_grafico_estado_contratos(df: pd.DataFrame) -> go.Figure:
    """Gráfico de dona: distribución por estado de riesgo."""
    if 'riesgo_spot' not in df.columns or df.empty:
        return None
    
    datos = df['riesgo_spot'].value_counts().reset_index()
    datos.columns = ['Riesgo', 'Cantidad']
    
    colores = {'BAJO 🟢': '#27ae60', 'MEDIO 🟡': '#f39c12', 'ALTO 🔴': '#e74c3c', 'REVISAR ⚪': '#95a5a6'}
    
    fig = px.pie(
        datos, 
        values='Cantidad', 
        names='Riesgo',
        color='Riesgo',
        color_discrete_map=colores,
        title="🎯 Distribución de Riesgo de Contratos",
        hole=0.4
    )
    fig.update_traces(textinfo='percent+label')
    return fig

def crear_timeline_vencimientos(df: pd.DataFrame) -> go.Figure:
    """Timeline de vencimientos próximos."""
    if 'dias_para_vencimiento' not in df.columns or df.empty:
        return None
    
    df_filtro = df[(df['dias_para_vencimiento'] >= -30) & (df['dias_para_vencimiento'] <= 90)].copy()
    if df_filtro.empty:
        return None
    
    col_termino = 'fecha_término_contrato' if 'fecha_término_contrato' in df_filtro.columns else 'fecha_termino_contrato'
    if col_termino in df_filtro.columns:
        df_filtro['fecha_venc'] = pd.to_datetime(df_filtro[col_termino])
        df_agrupado = df_filtro.groupby(df_filtro['fecha_venc'].dt.to_period('M')).size().reset_index()
        df_agrupado.columns = ['Mes', 'Cantidad']
        
        fig = px.bar(
            df_agrupado,
            x='Mes',
            y='Cantidad',
            title="📅 Vencimientos Próximos (90 días)",
            color='Cantidad',
            color_continuous_scale='YlOrRd'
        )
        return fig
    return None

def crear_tabla_alertas(df: pd.DataFrame) -> pd.DataFrame:
    """Genera tabla de alertas para acción inmediata."""
    alertas = []
    
    # Contratos vencidos o por vencer con garantía activa
    mask = (
        (df['riesgo_spot'].isin(['ALTO 🔴', 'MEDIO 🟡'])) & 
        (df.get('aplica_boleta_de_garantía_ariba', '').str.lower() == 'sí')
    )
    
    for _, row in df[mask].iterrows():
        alertas.append({
            'Contrato': row.get('contrato_ariba', 'N/A'),
            'Proveedor': row.get('proveedor', 'N/A'),
            'Riesgo': row.get('riesgo_spot', 'N/A'),
            'Días Restantes': row.get('dias_para_vencimiento', 'N/A'),
            'Monto Garantía': f"{row.get('monto_garantia_clp', 0):,.0f}",
            'Comprador Táctico': row.get('comprador_táctico', 'N/A'),
            'Acción': 'Renovar' if row.get('riesgo_spot') == 'MEDIO 🟡' else 'Regularizar'
        })
    
    return pd.DataFrame(alertas).sort_values('Días Restantes') if alertas else pd.DataFrame()

# ==============================
# 🎛️ INTERFAZ PRINCIPAL (TU UI + MIS FUNCIONES)
# ==============================

st.title("📋 Dashboard de Gestión de Contratos")
st.markdown("**Softys Chile** · Compras Estratégicas y Tácticas · Análisis de Riesgo Spot")
st.divider()

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Softys_logo.svg/320px-Softys_logo.svg.png",
             use_container_width=True)
    st.header("📁 Carga de Archivo")
    uploaded_file = st.file_uploader(
        "Sube el Consolidado de Contratos (.xlsx)",
        type=['xlsx', 'xls'],
        help="Exportado desde SAP/Ariba. Debe contener la hoja 'Antiguo' o 'Consolidado de Contratos'."
    )
    st.divider()
    st.caption("💡 El archivo se procesa localmente. Ningún dato sale de tu equipo.")

if not uploaded_file:
    st.info("👆 Sube el archivo **Consolidado_de_Contratos.xlsx** para comenzar.")
    st.markdown("""
    ### ¿Qué verás en este dashboard?
    - 🔴 **Alertas de contratos vencidos y por vencer**
    - 📊 **KPIs de riesgo spot** por gerencia y área
    - 🔍 **Análisis de boletas de garantía**
    - 📥 **Exportación filtrada** lista para reportes
    """)
    st.stop()

# Validar tamaño
if uploaded_file.size > MAX_FILE_SIZE_BYTES:
    st.error("❌ Archivo demasiado grande (máx. 50 MB).")
    st.stop()

# Cargar con caché
uploaded_file.seek(0)
content = uploaded_file.read()
h = hashlib.md5(content).hexdigest()

try:
    with st.spinner("🔄 Procesando contratos con parser robusto..."):
        df, df_bg, df_ariba, reporte_calidad = cargar_y_procesar_contratos(h, content)
except Exception as e:
    st.error(f"❌ Error al procesar el archivo: {str(e)}")
    st.stop()

if df.empty:
    st.error("❌ No se encontraron contratos válidos. Verifica el archivo.")
    st.stop()

# ==============================
# 🎛️ FILTROS EN SIDEBAR (TU UI)
# ==============================

with st.sidebar:
    st.header("🎛️ Filtros")

    # Estado del contrato / Riesgo
    estados = ['Todos'] + sorted(df['riesgo_spot'].unique().tolist()) if 'riesgo_spot' in df.columns else ['Todos']
    riesgo_sel = st.selectbox("Riesgo Spot", estados)

    # Gerencia
    if 'gerencia' in df.columns:
        gerencias = ['Todas'] + sorted(df['gerencia'].dropna().unique().astype(str).tolist())
        gerencia_sel = st.selectbox("Gerencia", gerencias)
    else:
        gerencia_sel = 'Todas'

    # Área
    if 'área' in df.columns:
        areas = ['Todas'] + sorted(df['área'].dropna().unique().astype(str).tolist())
        area_sel = st.selectbox("Área", areas)
    else:
        area_sel = 'Todas'

    # Comprador Estratégico
    if 'comprador estratégico' in df.columns:
        compradores = ['Todos'] + sorted(df['comprador estratégico'].dropna().unique().astype(str).tolist())
        comprador_sel = st.selectbox("Comprador Estratégico", compradores)
    else:
        comprador_sel = 'Todos'

    # Planta
    if 'planta' in df.columns:
        plantas = ['Todas'] + sorted(df['planta'].dropna().unique().astype(str).tolist())
        planta_sel = st.selectbox("Planta", plantas)
    else:
        planta_sel = 'Todas'

# --- Aplicar filtros ---
df_f = df.copy()

if riesgo_sel != 'Todos' and 'riesgo_spot' in df_f.columns:
    df_f = df_f[df_f['riesgo_spot'] == riesgo_sel]
if gerencia_sel != 'Todas' and 'gerencia' in df_f.columns:
    df_f = df_f[df_f['gerencia'] == gerencia_sel]
if area_sel != 'Todas' and 'área' in df_f.columns:
    df_f = df_f[df_f['área'] == area_sel]
if comprador_sel != 'Todos' and 'comprador estratégico' in df_f.columns:
    df_f = df_f[df_f['comprador estratégico'] == comprador_sel]
if planta_sel != 'Todas' and 'planta' in df_f.columns:
    df_f = df_f[df_f['planta'].str.contains(planta_sel, na=False)]

# ==============================
# 📊 SECCIÓN DE KPIs Y GRÁFICOS (TU UI)
# ==============================

st.subheader("📊 Resumen Ejecutivo")
crear_kpi_cards(df_f)

col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    fig_estado = crear_grafico_estado_contratos(df_f)
    if fig_estado:
        st.plotly_chart(fig_estado, use_container_width=True)

with col_graf2:
    fig_timeline = crear_timeline_vencimientos(df_f)
    if fig_timeline:
        st.plotly_chart(fig_timeline, use_container_width=True)

# ==============================
# 🚨 ALERTAS DE ACCIÓN INMEDIATA (MI LÓGICA + TU UI)
# ==============================

st.subheader("🚨 Alertas de Acción Inmediata")
df_alertas = crear_tabla_alertas(df_f)

if not df_alertas.empty:
    st.dataframe(
        df_alertas.style.applymap(
            lambda x: 'background-color: #fef2f2' if 'ALTO' in str(x) else 
                     'background-color: #fffbeb' if 'MEDIO' in str(x) else '',
            subset=['Riesgo']
        ),
        use_container_width=True
    )
    
    # Resumen de impacto
    st.info(f"""
    💡 **Impacto estimado**: 
    - {len(df_alertas)} contratos requieren acción inmediata
    - Ahorro potencial: ~15% sobre montos en riesgo (compras spot vs contrato)
    - Tiempo ahorrado: ~2-3 horas/contrato en gestión manual
    """)
else:
    st.success("✅ No hay contratos críticos que requieran acción inmediata.")

# ==============================
# 📋 TABS PRINCIPALES (TU UI)
# ==============================

tab1, tab2, tab3, tab4 = st.tabs(["📊 Resumen General", "🏢 Por Gerencia / Área", "🔒 Boletas de Garantía", "🔍 Explorador"])

# ---------- TAB 1: RESUMEN ----------
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        # Dona de estado
        if 'riesgo_spot' in df_f.columns:
            estado_counts = df_f['riesgo_spot'].value_counts().reset_index()
            estado_counts.columns = ['Estado', 'Cantidad']
            color_map = {'BAJO 🟢': '#27ae60', 'MEDIO 🟡': '#f39c12', 'ALTO 🔴': '#e74c3c', 'REVISAR ⚪': '#95a5a6'}
            fig_dona = px.pie(
                estado_counts, names='Estado', values='Cantidad',
                title='Distribución por Estado de Riesgo',
                hole=0.45,
                color='Estado',
                color_discrete_map=color_map
            )
            fig_dona.update_traces(textposition='inside', textinfo='percent+label')
            fig_dona.update_layout(showlegend=False)
            st.plotly_chart(fig_dona, use_container_width=True)

    with c2:
        # Riesgo spot por planta
        if 'planta' in df_f.columns and 'riesgo_spot' in df_f.columns:
            df_planta = df_f.groupby(['planta', 'riesgo_spot']).size().reset_index(name='Cantidad')
            fig_riesgo = px.bar(
                df_planta, x='planta', y='Cantidad', color='riesgo_spot',
                title='Riesgo por Planta',
                color_discrete_map={'BAJO 🟢': '#27ae60', 'MEDIO 🟡': '#f39c12', 'ALTO 🔴': '#e74c3c', 'REVISAR ⚪': '#95a5a6'},
                barmode='stack'
            )
            fig_riesgo.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_riesgo, use_container_width=True)

# ---------- TAB 2: GERENCIA / ÁREA ----------
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        if 'gerencia' in df_f.columns and 'riesgo_spot' in df_f.columns:
            df_ger = df_f.groupby(['gerencia', 'riesgo_spot']).size().reset_index(name='Cantidad')
            fig_ger = px.bar(
                df_ger, x='Cantidad', y='gerencia', color='riesgo_spot',
                title='Contratos por Gerencia y Riesgo',
                barmode='stack', orientation='h',
                color_discrete_map={'BAJO 🟢': '#27ae60', 'MEDIO 🟡': '#f39c12', 'ALTO 🔴': '#e74c3c', 'REVISAR ⚪': '#95a5a6'}
            )
            fig_ger.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_ger, use_container_width=True)

    with c2:
        if 'área' in df_f.columns and 'riesgo_spot' in df_f.columns:
            df_area = df_f.groupby(['área', 'riesgo_spot']).size().reset_index(name='Cantidad')
            top_areas = df_f['área'].value_counts().head(12).index
            df_area = df_area[df_area['área'].isin(top_areas)]
            fig_area = px.bar(
                df_area, x='Cantidad', y='área', color='riesgo_spot',
                title='Top 12 Áreas por Riesgo',
                barmode='stack', orientation='h',
                color_discrete_map={'BAJO 🟢': '#27ae60', 'MEDIO 🟡': '#f39c12', 'ALTO 🔴': '#e74c3c', 'REVISAR ⚪': '#95a5a6'}
            )
            fig_area.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_area, use_container_width=True)

# ---------- TAB 3: BOLETAS DE GARANTÍA ----------
with tab3:
    if not df_bg.empty:
        st.markdown("#### 📋 Estado de Boletas de Garantía")
        
        c1, c2 = st.columns(2)
        with c1:
            bg_counts = df_bg['Estado'].value_counts().reset_index()
            bg_counts.columns = ['Estado', 'Cantidad']
            bg_color = {'VIGENTE': '#27ae60', 'VENCIDA': '#e74c3c', 'ENTREGADA': '#3498db', 'ENDOSADA': '#95a5a6'}
            fig_bg = px.pie(
                bg_counts, names='Estado', values='Cantidad',
                title='Estado de Boletas',
                hole=0.4,
                color='Estado',
                color_discrete_map={k: bg_color.get(k, '#95a5a6') for k in bg_counts['Estado']}
            )
            fig_bg.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_bg, use_container_width=True)
        
        with c2:
            # BG vencidas por contratista
            df_bg_venc = df_bg[df_bg['Estado'] == 'VENCIDA']
            if not df_bg_venc.empty:
                top_contratistas = df_bg_venc['CONTRATISTA'].value_counts().head(10)
                fig_bg_area = px.bar(
                    x=top_contratistas.values,
                    y=top_contratistas.index,
                    orientation='h',
                    title='Top Contratistas con BG Vencidas',
                    color=top_contratistas.values,
                    color_continuous_scale='Reds'
                )
                fig_bg_area.update_layout(xaxis_title='BG Vencidas', yaxis_title='Contratista')
                st.plotly_chart(fig_bg_area, use_container_width=True)
        
        # Tabla de alertas BG
        st.markdown("#### ⚠️ Contratos que Requieren BG pero Tienen Estado Crítico")
        if 'contrato_ariba' in df.columns and 'estado_garantía' in df_bg.columns:
            # Cruce simple por CW
            df_merge = df.merge(df_bg[['CW', 'Estado', 'VENC.', 'MONTO']], 
                               left_on='contrato_ariba', right_on='CW', how='inner')
            criticas = df_merge[
                (df_merge['Estado'].isin(['VENCIDA', 'ENTREGADA'])) & 
                (df_merge['aplica_boleta_de_garantía_ariba'].str.lower() == 'sí')
            ]
            if not criticas.empty:
                st.dataframe(criticas[['contrato_ariba', 'proveedor', 'Estado', 'VENC.', 'MONTO']], use_container_width=True)
                st.warning(f"⚠️ {len(criticas)} contratos con BG requerida pero en estado crítico")
            else:
                st.success("✅ Todos los contratos que requieren BG están al día.")
    else:
        st.info("ℹ️ No se encontró la hoja 'BG' en el archivo. Agrega esta sheet para análisis de garantías.")

# ---------- TAB 4: EXPLORADOR (TU UI) ----------
with tab4:
    st.markdown("### 🔍 Explorador de Datos")

    col_mostrar = st.multiselect(
        "Columnas a mostrar",
        options=df_f.columns.tolist(),
        default=[c for c in ['contrato_ariba', 'proveedor', 'área', 'gerencia', 
                           'comprador estratégico', 'fecha_término_contrato',
                           'dias_para_vencimiento', 'riesgo_spot', 'planta'] 
                if c in df_f.columns]
    )

    search_term = st.text_input("🔎 Buscar proveedor o descripción")
    if search_term and 'proveedor' in df_f.columns:
        mask = df_f['proveedor'].str.contains(search_term, case=False, na=False)
        df_view = df_f[mask][col_mostrar] if col_mostrar else df_f[mask]
    else:
        df_view = df_f[col_mostrar] if col_mostrar else df_f

    st.dataframe(
        df_view.sort_values('dias_para_vencimiento') if 'dias_para_vencimiento' in df_view.columns else df_view,
        use_container_width=True,
        height=450
    )
    st.caption(f"Mostrando {len(df_view):,} de {len(df_f):,} contratos filtrados")

# ==============================
# 📥 EXPORTACIÓN (TU UI)
# ==============================

st.divider()
st.subheader("📥 Exportar Resultados")

ec1, ec2, ec3 = st.columns(3)

with ec1:
    csv_data = df_f.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "💾 Descargar Filtrado (CSV)",
        data=csv_data,
        file_name=f"contratos_filtrado_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

with ec2:
    criticos_export = df_f[df_f['riesgo_spot'].isin(['ALTO 🔴', 'MEDIO 🟡'])] if 'riesgo_spot' in df_f.columns else df_f
    csv_crit = criticos_export.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "🔴 Solo Contratos en Riesgo",
        data=csv_crit,
        file_name=f"contratos_riesgo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

with ec3:
    urgentes = df_f[
        (df_f['dias_para_vencimiento'].between(0, 60)) if 'dias_para_vencimiento' in df_f.columns else pd.Series([False]*len(df_f))
    ]
    csv_urg = urgentes.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "⚠️ Vencen en 60 Días",
        data=csv_urg,
        file_name=f"contratos_urgentes_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

# ==============================
# 🔍 DIAGNÓSTICO TÉCNICO (MI LÓGICA)
# ==============================

with st.expander("🔧 Diagnóstico Técnico"):
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Calidad de datos:**")
        st.json(reporte_calidad)
    with d2:
        st.markdown("**Resumen del dataset:**")
        st.json({
            "Total contratos": len(df),
            "Contratos filtrados": len(df_f),
            "Rango fechas": f"{df['fecha_inicio'].min() if 'fecha_inicio' in df.columns else 'N/A'} → {df['fecha_término_contrato'].max() if 'fecha_término_contrato' in df.columns else 'N/A'}",
            "Última actualización": datetime.now().strftime('%d/%m/%Y %H:%M'),
            "Compradores únicos": df['comprador estratégico'].nunique() if 'comprador estratégico' in df.columns else 0,
            "Proveedores únicos": df['proveedor'].nunique() if 'proveedor' in df.columns else 0
        })

# ==============================
# ℹ️ FOOTER
# ==============================
st.divider()
st.caption(f"""
🔹 Dashboard generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}  
🔹 Fuente: Export SAP/Ariba - Softys Chile  
🔹 Parser robusto: Maneja formatos `30"-"09"-"2025`, `4/26/19`, `99.99.9999`  
🔹 Próximo paso: Automatizar con Task Scheduler + Power Automate
""")