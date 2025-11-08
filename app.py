# app.py
"""
App Streamlit: clasificación ABC vs AHP para Shipping y Labor.
Mantener nombres de pestañas: "Shipping Detail Report" y "Labor Activity Report".
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import json2html

from ahp import compute_abc, compute_ahp, compute_similarity_metrics, compute_summary

st.set_page_config(layout="wide", page_title="ABC vs AHP Dashboard")

st.title("Clasificación ABC vs AHP — Shipping & Labor")
st.markdown("**Instrucciones:** Subir un archivo Excel que contenga *exactamente* las hojas `Shipping Detail Report` y `Labor Activity Report` (no renombres las pestañas).")

uploaded = st.file_uploader("Sube archivo Excel (.xlsx) con ambas hojas", type=['xlsx'])

if not uploaded:
    st.info("Sube un archivo Excel para empezar. Asegúrate de que las hojas se llamen exactamente 'Shipping Detail Report' y 'Labor Activity Report'.")
    st.stop()

# Cargar archivos
try:
    xls = pd.ExcelFile(uploaded)
    available_sheets = xls.sheet_names
    if "Shipping Detail Report" not in available_sheets or "Labor Activity Report" not in available_sheets:
        st.error("El archivo no contiene una o ambas hojas requeridas: 'Shipping Detail Report' y 'Labor Activity Report'. Verifica el archivo.")
        st.stop()
    df_ship_raw = pd.read_excel(xls, sheet_name="Shipping Detail Report")
    df_labor_raw = pd.read_excel(xls, sheet_name="Labor Activity Report")
except Exception as e:
    st.exception(e)
    st.stop()

st.sidebar.header("Configuración general")
view_mode = st.sidebar.selectbox("Ver dataset", options=['Shipping','Labor'])
# Preview y selección de columna fecha
st.subheader("Preview y selección de columna fecha")

col1, col2 = st.columns(2)
with col1:
    st.write("**Shipping - head()**")
    st.dataframe(df_ship_raw.head())
    ship_date_col = st.selectbox("Selecciona columna fecha (Shipping)", options=list(df_ship_raw.columns), index=0, key='ship_date')
with col2:
    st.write("**Labor - head()**")
    st.dataframe(df_labor_raw.head())
    labor_date_col = st.selectbox("Selecciona columna fecha (Labor)", options=list(df_labor_raw.columns), index=0, key='labor_date')

# Convertir a datetime y pedir rango (aplicado a ambos)
df_ship_raw[ship_date_col] = pd.to_datetime(df_ship_raw[ship_date_col], errors='coerce')
df_labor_raw[labor_date_col] = pd.to_datetime(df_labor_raw[labor_date_col], errors='coerce')

min_date = min(df_ship_raw[ship_date_col].min(), df_labor_raw[labor_date_col].min())
max_date = max(df_ship_raw[ship_date_col].max(), df_labor_raw[labor_date_col].max())

st.sidebar.subheader("Filtrar rango de fechas (aplicado a ambos datasets)")
date_range = st.sidebar.date_input("Rango fechas", value=[min_date.date(), max_date.date()], min_value=min_date.date(), max_value=max_date.date())
start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# Aplicar filtro a ambos datasets
df_ship = df_ship_raw[(df_ship_raw[ship_date_col] >= start) & (df_ship_raw[ship_date_col] <= end)].copy()
df_labor = df_labor_raw[(df_labor_raw[labor_date_col] >= start) & (df_labor_raw[labor_date_col] <= end)].copy()

st.sidebar.write(f"Shipping filtrado: {df_ship.shape[0]} filas | Labor filtrado: {df_labor.shape[0]} filas")

# ---------------------------
# Parámetros ABC (umbral editable)
# ---------------------------
st.sidebar.subheader("Umbrales ABC (cortes acumulativos en %)")
default_cuts = [80, 90]
cut_a = st.sidebar.slider("A hasta (%)", min_value=1, max_value=99, value=default_cuts[0])
cut_b = st.sidebar.slider("B hasta (%)", min_value=cut_a+1, max_value=100, value=default_cuts[1])
cuts = [cut_a, cut_b]

# ---------------------------
# Selección columnas SKU y agregación
# ---------------------------
st.sidebar.subheader("Columnas clave y agregación")
sku_col_ship = st.sidebar.text_input("Nombre de columna SKU (Shipping)", value="SKU")
qty_col_ship = st.sidebar.text_input("Nombre Qty Shipped (Shipping)", value="Qty Shipped")
weight_col_ship = st.sidebar.text_input("Nombre Weight [Kg] (Shipping)", value="Weight [Kg]")
boxes_col_ship = st.sidebar.text_input("Nombre Boxes (Shipping)", value="Boxes")

sku_col_labor = st.sidebar.text_input("Nombre de columna SKU (Labor)", value="SKU")
# Sugerir métricas de labor (por ejemplo Hours, Headcount) - el usuario selecciona las columnas que quiera usar para AHP
# No forzamos nombres específicos para labor; permitimos seleccionar más abajo.

# ---------------------------
# Selección de variables para AHP (features)
# ---------------------------
st.sidebar.subheader("Selección de variables para AHP")
if view_mode == 'Shipping':
    # Mostrar columnas sugeridas para usar en AHP
    candidate_features_ship = [qty_col_ship, weight_col_ship, boxes_col_ship]
    use_features = st.sidebar.multiselect("Selecciona variables para AHP (Shipping)", options=candidate_features_ship, default=candidate_features_ship)
else:
    # Para labor, permitir seleccionar columnas del dataframe lab
    candidate_features_labor = list(df_labor.columns)
    use_features = st.sidebar.multiselect("Selecciona variables para AHP (Labor)", options=candidate_features_labor, default=candidate_features_labor[:3])

# ---------------------------
# Panel AHP: subir imagen explicativa y editar matriz
# ---------------------------
st.header("Panel AHP")
st.markdown("Sube una imagen explicativa (opcional) que se mostrará encima de la matriz de comparaciones. Luego edita la matriz o los pesos directamente.")

uploaded_img = st.file_uploader("Sube imagen explicativa (PNG/JPG)", type=['png','jpg','jpeg'])
if uploaded_img:
    st.image(uploaded_img, caption="Imagen explicativa AHP", use_column_width=True)

# Crear interfaz para editar matriz de comparaciones entre criterios (features)
st.subheader("Editar matriz de comparaciones (AHP)")
st.markdown("Usa valores 1,3,5,7,9 y recíprocos (1/3, 1/5...). Puedes editar cada par. Alternativa: editar pesos directos.")

# Preparar estructura de comparaciones para ahpy: dict of dicts
features = use_features.copy()
if len(features) < 2:
    st.warning("Selecciona al menos 2 features para ejecutar AHP.")
    st.stop()

# Construir dataframe triangular para edición
pairs = []
for i in range(len(features)):
    for j in range(i+1, len(features)):
        pairs.append({'feature_i': features[i], 'feature_j': features[j], 'value': 1.0})

pairs_df = pd.DataFrame(pairs)
# Mostrar editor del dataframe
edited_pairs = st.data_editor(pairs_df, num_rows="dynamic", key="ahp_pairs_editor")


# Convertir a comparisons dict para ahpy
comparisons = {}
for _, row in edited_pairs.iterrows():
    i = row['feature_i']
    j = row['feature_j']
    val = float(row['value']) if pd.notna(row['value']) else 1.0
    comparisons[(i,j)] = val

# Opción alternativa: editar pesos directos
st.markdown("**Opción:** editar pesos directos (sobreescribe la matriz si se usan).")
weights_manual = {}
use_manual_weights = st.checkbox("Usar pesos manuales (si activo, la matriz se ignorará para pesos)")
if use_manual_weights:
    col1, col2 = st.columns(2)
    for f in features:
        weights_manual[f] = col1.number_input(f"Peso {f}", min_value=0.0, value=1.0, step=0.1, key=f"w_{f}")
    # Normalizar pesos
    total_w = sum(weights_manual.values())
    if total_w == 0:
        # evitar dividir por cero
        for k in weights_manual:
            weights_manual[k] = 1.0/len(weights_manual)
    else:
        for k in weights_manual:
            weights_manual[k] = weights_manual[k]/total_w

# ---------------------------
# Botón ejecutar
# ---------------------------
if st.button("Ejecutar clasificación y métricas"):
    # Elegir dataset
    if view_mode == 'Shipping':
        df_use = df_ship.copy()
        sku_col = sku_col_ship
        aggregations = {'Qty Shipped': qty_col_ship, 'Weight [Kg]': weight_col_ship, 'Boxes': boxes_col_ship}
    else:
        df_use = df_labor.copy()
        sku_col = sku_col_labor
        # Asumir que user seleccionó features pertinentes; si no existen, compute_summary rellenará con ceros
        aggregations = {'Qty Shipped': use_features[0] if len(use_features)>0 else use_features[0], 'Weight [Kg]': use_features[1] if len(use_features)>1 else use_features[0], 'Boxes': use_features[2] if len(use_features)>2 else use_features[0]}

    # Calcular summary por SKU
    summary = compute_summary(df_use, sku_col=sku_col, aggregations=aggregations)

    # ABC clásico
    abc_df = compute_abc(summary, qty_col='Qty Shipped', cuts=cuts)

    # AHP: construir comparisons dict (si usan pesos manuales, creamos comparisons que reflejen esos pesos)
    if use_manual_weights:
        # Si el usuario quiere usar pesos manuales, construiremos un faux-comparisons dict que refleje ratios de pesos
        comp_manual = {}
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                a = features[i]; b = features[j]
                val = weights_manual[a] / weights_manual[b] if weights_manual[b] != 0 else 1.0
                comp_manual.setdefault(a, {})[b] = float(val)
        comparisons_to_use = comp_manual
    else:
        comparisons_to_use = comparisons

    # Ejecutar AHP y obtener summary ahp
    ahp_summary, criteria = compute_ahp(summary, features=features, comparisons_dict=comparisons_to_use, cuts=cuts)

    # Unir ABC clásico y AHP en un solo df comparativo (usar SKU como key)
    merged = pd.merge(abc_df[['SKU','Qty Shipped','Weight [Kg]','Boxes','cum%','ABC_class']], 
                      ahp_summary[['SKU','AHP_score','cum_AHP%','AHP_class'] + [f'{f}_norm' for f in features]],
                      on='SKU', how='outer').fillna(0)

    st.header("Resultados comparativos")
    st.subheader("Tabla de resumen (por SKU)")
    st.dataframe(merged)

    # Mostrar conteos por clase
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("ABC - A count", int((merged['ABC_class']=='A').sum()))
        st.metric("AHP - A count", int((merged['AHP_class']=='A').sum()))
    with colB:
        st.metric("ABC - B count", int((merged['ABC_class']=='B').sum()))
        st.metric("AHP - B count", int((merged['AHP_class']=='B').sum()))
    with colC:
        st.metric("ABC - C count", int((merged['ABC_class']=='C').sum()))
        st.metric("AHP - C count", int((merged['AHP_class']=='C').sum()))

    # Mostrar reporte ahpy si estuvo bien construido
    if criteria is not None:
        st.subheader("Reporte AHP (ahpy)")
        #try:
        st.html(json2html.json2html.convert(criteria.report()))
        # except Exception:
        #     st.write("No se pudo renderizar report() de ahpy; mostrando pesos y consistency_ratio si están disponibles.")
        #     st.write("Pesos (target_weights):", criteria.target_weights)
        #     try:
        #         st.write("Consistency ratio:", criteria.consistency_ratio)
        #     except Exception:
        #         st.write("Consistency ratio no disponible.")

    else:
        st.warning("No se generó un objeto 'criteria' válido. Se usaron pesos iguales o manuales.")

    # Métricas de similitud
    st.subheader("Métricas de similitud por clase y globales")
    # Para métricas, usar columnas numéricas escogidas (features)
    merged_for_metrics = merged.copy()
    # Asegurarse de tener AHP_score y las features normalizadas
    metrics_df, global_indices = compute_similarity_metrics(merged_for_metrics, features=[f'{f}_norm' for f in features], class_col='AHP_class')
    st.dataframe(metrics_df)
    st.write("Índices globales:", global_indices)

    # Boxplots comparativos: Qty Shipped / Weight [Kg] / Boxes vs classes (AHP_class y ABC_class)
    st.subheader("Boxplots comparativos")
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.markdown("**Qty Shipped por AHP_class (Plotly interactivo)**")
        try:
            fig_px = px.box(merged, x='AHP_class', y='Qty Shipped', points="all", title="Qty Shipped vs AHP_class")
            st.plotly_chart(fig_px, use_container_width=True)
        except Exception as e:
            st.write("No se pudo generar Plotly:", e)
    with fig_col2:
        st.markdown("**Qty Shipped por ABC_class (Plotly interactivo)**")
        try:
            fig_px2 = px.box(merged, x='ABC_class', y='Qty Shipped', points="all", title="Qty Shipped vs ABC_class")
            st.plotly_chart(fig_px2, use_container_width=True)
        except Exception as e:
            st.write("No se pudo generar Plotly:", e)

    # Generar también gráficos estáticos con matplotlib/seaborn (por compatibilidad con el ejemplo)
    st.markdown("**Gráficos estáticos (matplotlib/seaborn)**")
    fig, axes = plt.subplots(1,3, figsize=(18,5))
    sns.boxplot(data=merged, x='AHP_class', y='Qty Shipped', ax=axes[0])
    axes[0].set_title("Qty Shipped vs AHP_class")
    sns.boxplot(data=merged, x='AHP_class', y='Weight [Kg]', ax=axes[1])
    axes[1].set_title("Weight [Kg] vs AHP_class")
    sns.boxplot(data=merged, x='AHP_class', y='Boxes', ax=axes[2])
    axes[2].set_title("Boxes vs AHP_class")
    st.pyplot(fig)

    # Botón de descarga
    st.subheader("Descargar resumen")
    to_download = merged.copy()
    csv = to_download.to_csv(index=False).encode('utf-8')
    st.download_button(label="Descargar CSV", data=csv, file_name=f"{view_mode}_summary.csv", mime='text/csv')

    # Mensajes de error/advertencia
    # Consistencia AHP
    try:
        if criteria is not None:
            cr = getattr(criteria, 'consistency_ratio', None)
            if cr is not None:
                if cr > 0.1:
                    st.warning(f"Consistency ratio AHP = {cr:.3f} > 0.1. Revisa la matriz de comparaciones; puede no ser consistente.")
                else:
                    st.success(f"Consistency ratio AHP = {cr:.3f}. Consistencia aceptable.")
    except Exception:
        st.info("No fue posible obtener la consistency_ratio del objeto ahpy.")

    st.success("Ejecución completada.")

# Fin del botón ejecutar
