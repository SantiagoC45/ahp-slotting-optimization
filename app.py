# app.py
"""
App Streamlit: clasificación ABC vs AHP para Shipping y Labor.
Mantener nombres de pestañas: "Shipping Detail Report" y "Labor Activity Report".
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import json2html

from ahp import compute_abc, compute_ahp, compute_similarity_metrics, compute_summary


# --- CACHE Y FUNCIONES AUXILIARES ---
@st.cache_data
def load_excel(uploaded):
    """Carga y cachea las dos hojas del Excel."""
    xls = pd.ExcelFile(uploaded)
    df_ship_raw = pd.read_excel(xls, sheet_name="Shipping Detail Report")
    df_labor_raw = pd.read_excel(xls, sheet_name="Labor Activity Report")
    return df_ship_raw, df_labor_raw

@st.cache_data
def preprocess_data(df, date_col, start, end):
    """Convierte fechas y aplica filtro temporal."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    return df[(df[date_col] >= start) & (df[date_col] <= end)].copy()


st.set_page_config(layout="wide", page_title="ABC vs AHP Dashboard")

st.title("Clasificación ABC vs AHP — Shipping & Labor")
st.markdown("**Instrucciones:** Subir un archivo Excel que contenga *exactamente* las hojas `Shipping Detail Report` y `Labor Activity Report` (no renombres las pestañas).")

uploaded = st.file_uploader("Sube archivo Excel (.xlsx) con ambas hojas", type=['xlsx'])

if not uploaded:
    st.info("Sube un archivo Excel para empezar. Asegúrate de que las hojas se llamen exactamente 'Shipping Detail Report' y 'Labor Activity Report'.")
    st.stop()

# Cargar archivos (cacheado)
try:
    df_ship_raw, df_labor_raw = load_excel(uploaded)
except Exception as e:
    st.exception(e)
    st.stop()


st.sidebar.header("Configuración general")
view_mode = st.sidebar.selectbox("Ver dataset", options=['Shipping','Labor'])
# Preview y selección de columna fecha
st.subheader("Preview y selección de columna fecha")

col1, col2 = st.columns(2)
with col1:
    st.write("**Shipping**")
    st.dataframe(df_ship_raw.head())
    ship_date_col = st.selectbox("Selecciona columna fecha (Shipping)", options=list(df_ship_raw.columns), index=0, key='ship_date')
with col2:
    st.write("**Labor**")
    st.dataframe(df_labor_raw.head())
    labor_date_col = st.selectbox("Selecciona columna fecha (Labor)", options=list(df_labor_raw.columns), index=0, key='labor_date')

# Convertir a datetime solo una vez
df_ship_raw[ship_date_col] = pd.to_datetime(df_ship_raw[ship_date_col], errors='coerce')
df_labor_raw[labor_date_col] = pd.to_datetime(df_labor_raw[labor_date_col], errors='coerce')

# Calcular fechas mín/máx
min_date = min(df_ship_raw[ship_date_col].min(), df_labor_raw[labor_date_col].min())
max_date = max(df_ship_raw[ship_date_col].max(), df_labor_raw[labor_date_col].max())

st.sidebar.subheader("Filtrar rango de fechas (aplicado a ambos datasets)")
date_range = st.sidebar.date_input(
    "Rango fechas",
    value=[min_date.date(), max_date.date()],
    min_value=min_date.date(),
    max_value=max_date.date()
)
start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# Aplicar filtro (cacheado)
df_ship = preprocess_data(df_ship_raw, ship_date_col, start, end)
df_labor = preprocess_data(df_labor_raw, labor_date_col, start, end)


st.sidebar.write(f"Shipping filtrado: {df_ship.shape[0]} filas \nLabor filtrado: {df_labor.shape[0]} filas")

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
#st.sidebar.subheader("Columnas clave y agregación")
sku_col_ship = "SKU"
qty_col_ship = "Qty Shipped"
weight_col_ship = "Weight [Kg]"
boxes_col_ship = "Boxes"

sku_col_labor = "SKU"
weight_col_labor = "WEIGHT [Kg]"
qty_col_labor = "Pick Unit"

# ---------------------------
# Selección de variables para AHP (features)
# ---------------------------
if view_mode == 'Shipping':
    df_use = df_ship
    candidate_features_ship = [qty_col_ship, weight_col_ship, boxes_col_ship]
    default_features = [f for f in candidate_features_ship if f in df_use.columns]
else:
    df_use = df_labor
    candidate_features_labor = [weight_col_labor, qty_col_labor]
    default_features = [f for f in candidate_features_labor if f in df_use.columns]

# Detectar columnas numéricas
numeric_cols = df_use.select_dtypes(include=['int64', 'float64']).columns.tolist()

st.sidebar.markdown(f"### ⚙️ Selección de variables para AHP ({view_mode})")

if not numeric_cols:
    st.sidebar.warning(f"No se encontraron columnas numéricas en el dataset de {view_mode}.")
    use_features = []
else:
    use_features = st.sidebar.multiselect(
        f"Selecciona variables numéricas para AHP ({view_mode})",
        options=numeric_cols,
        default=default_features if default_features else numeric_cols
    )

# ---------------------------
# Panel AHP: subir imagen explicativa y editar matriz
# ---------------------------
st.header("Panel AHP")
st.markdown("### Escala de comparación por pares (Método AHP)")

st.markdown("""
Esta tabla muestra la **escala de intensidad de importancia** utilizada en el método de comparación por pares de **Analytic Hierarchy Process (AHP)**.
Se emplea para evaluar la importancia relativa entre dos criterios.

| Intensidad de Importancia | Definición | Explicación |
|:---------------------------|:-----------|:-------------|
| **1** | Igual importancia | Las dos actividades contribuyen de igual forma al objetivo. |
| **2** | Importancia débil o ligera | Preferencia muy leve de una actividad sobre otra. |
| **3** | Importancia moderada | La experiencia y el juicio favorecen ligeramente a una actividad sobre otra. |
| **4** | Moderada más | Valor intermedio entre moderada y fuerte. |
| **5** | Importancia fuerte | La experiencia y el juicio favorecen claramente a una actividad sobre otra. |
| **6** | Fuerte más | Valor intermedio entre fuerte y muy fuerte. |
| **7** | Importancia muy fuerte o demostrada | Una actividad es claramente más importante; su superioridad se demuestra en la práctica. |
| **8** | Muy, muy fuerte | Valor intermedio entre muy fuerte y extrema. |
| **9** | Importancia extrema | La evidencia a favor de una actividad sobre otra es del más alto grado posible. |
""")

st.info("💡 Usa esta tabla como guía para llenar la **matriz de comparaciones** o definir los **pesos directos** de los criterios.")

# Crear interfaz para editar matriz de comparaciones entre criterios (features)
st.subheader("Editar matriz de comparaciones (AHP)")
st.markdown("Usa valores 1,3,5,7,9. Puedes editar cada par. Alternativa: editar pesos directos.")

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

    # Ingreso de pesos manuales
    for f in features:
        weights_manual[f] = col1.number_input(
            f"Peso {f}",
            min_value=0.0,
            value=0.0,
            step=0.05,
            key=f"w_{f}"
        )

    # Calcular suma total
    total_w = sum(weights_manual.values())

    # Mostrar barra de progreso visual
    st.markdown(f"**Suma actual de pesos:** {total_w:.3f}")
    progress_value = min(total_w, 1.0)  # evita pasar de 100%
    st.progress(progress_value)

    # Verificar si la suma es válida
    if abs(total_w - 1.0) > 1e-6:
        st.warning(f"La suma de los pesos ({total_w:.2f}) debe ser exactamente 1. "
                   "Ajusta los valores antes de continuar.")
        st.stop()
    else:
        # Normalizar pesos (por si hay redondeos)
        for k in weights_manual:
            weights_manual[k] = weights_manual[k] / total_w
        st.success("✅ Pesos válidos: la suma es 1.")


# ---------------------------
# Botón ejecutar
# ---------------------------
if st.button("Ejecutar clasificación y métricas"):
    if view_mode == 'Shipping':
        df_use = df_ship.copy()
        sku_col = sku_col_ship
    else:
        df_use = df_labor.copy()
        sku_col = sku_col_labor

    # Construir 'aggregations' dinámicamente en base a las features seleccionadas
    if use_features and len(use_features) > 0:
        aggregations = {col: col for col in use_features}
    else:
        # Fallback: usar columnas por defecto según el modo
        if view_mode == 'Shipping':
            aggregations = {
                'Qty Shipped': qty_col_ship,
                'Weight [Kg]': weight_col_ship,
                'Boxes': boxes_col_ship
            }
        else:
            aggregations = {
                'Weight [Kg]': weight_col_labor,
                'Pick Unit': qty_col_labor
            }

    # Calcular summary por SKU
    summary = compute_summary(df_use, mode=view_mode, sku_col=sku_col, aggregations=aggregations)

    # ABC clásico
    if view_mode == 'Shipping':
        qty_col = 'Qty Shipped'
    else:  # labor
        qty_col = 'Pick Unit'

    abc_df = compute_abc(summary, qty_col=qty_col, cuts=cuts)


    # AHP: construir comparisons dict (si usan pesos manuales, creamos comparisons que reflejen esos pesos)
    if use_manual_weights:
        ahp_summary, criteria = compute_ahp(summary, features=features, comparisons_dict=comparisons, cuts=cuts, w=weights_manual)
    else:
        comparisons_to_use = comparisons
        ahp_summary, criteria = compute_ahp(summary, features=features, comparisons_dict=comparisons, cuts=cuts)

    # Unir ABC clásico y AHP en un solo df comparativo (usar SKU como key)
    
    abc_cols = ['SKU'] + use_features + ['cum%', 'ABC_class']

    # Asegurar que solo se usen columnas que existen
    abc_cols = [c for c in abc_cols if c in abc_df.columns]

    # Asegurar que las columnas AHP estén disponibles
    ahp_cols = ['SKU', 'AHP_score', 'cum_AHP%', 'AHP_class'] + [f'{f}_norm' for f in features if f'{f}_norm' in ahp_summary.columns]

    # Hacer el merge
    merged = (
        pd.merge(
            abc_df[abc_cols],
            ahp_summary[ahp_cols],
            on='SKU',
            how='outer'
        )
        .fillna(0)
    )

    # Mostrar reporte ahpy si estuvo bien construido
    if criteria is not None:
        st.subheader("Reporte AHP (ahpy)")
        try:
            if criteria is not None:
                # Obtener Consistency Ratio
                cr = getattr(criteria, 'consistency_ratio', None)

                # Mostrar el CR
                if cr is not None:
                    if cr > 0.1:
                        st.warning(f"Consistency ratio AHP = {cr:.3f} > 0.1. Revisa la matriz de comparaciones; puede no ser consistente.")
                    else:
                        st.success(f"Consistency ratio AHP = {cr:.3f}. Consistencia aceptable.")
                else:
                    st.info("No se encontró el atributo 'consistency_ratio' en el objeto AHP.")

                # Mostrar los pesos obtenidos por criterio
                try:
                    weights = criteria.target_weights
                    if weights:
                        st.subheader("Pesos obtenidos por criterio (AHP)")
                        weights_df = pd.DataFrame(list(weights.items()), columns=["Criterio", "Peso"])
                        weights_df["Peso"] = weights_df["Peso"].round(4)
                        st.table(weights_df)
                    else:
                        st.warning("No se encontraron pesos en el objeto 'criteria'.")
                except Exception as e:
                    st.warning(f"No fue posible obtener los pesos del objeto AHP: {e}")

            else:
                st.info("No se generó un objeto 'criteria' válido para el cálculo de AHP.")
        except Exception as e:
            st.info(f"No fue posible obtener la consistency_ratio del objeto ahpy: {e}")

    else:
        st.warning("No se generó un objeto 'criteria' válido. Se usaron pesos manuales.")

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

    # Métricas de similitud
    st.subheader("Métricas de similitud por clase y globales")
    # Para métricas, usar columnas numéricas escogidas (features)
    merged_for_metrics = merged.copy()
    # Asegurarse de tener AHP_score y las features normalizadas
    metrics_df, global_indices = compute_similarity_metrics(merged_for_metrics, features=[f'{f}_norm' for f in features], class_col='AHP_class')
    st.dataframe(metrics_df)
    st.write("Índices globales:", global_indices)

    # Boxplots comparativos dinámicos basados en las features elegidas
    st.subheader("Boxplots comparativos")

    # Asegurar que existan features seleccionadas
    if not features:
        st.warning("Por favor selecciona al menos una feature para graficar.")
    else:
        # Limitar a features que realmente existen en el DataFrame merged
        plot_features = [f for f in features if f in merged.columns]

        if not plot_features:
            st.warning("Ninguna de las features seleccionadas existe en los datos combinados.")
        else:
            # Plotly interactivos
            st.markdown("### Plotly interactivos")
            for feature in plot_features:
                fig_col1, fig_col2 = st.columns(2)

                # --- AHP_class ---
                # --- AHP_class ---
                # --- AHP_class ---
                with fig_col1:
                    try:
                        # Crear el boxplot base
                        fig_px = px.box(
                            merged,
                            x='AHP_class',
                            y=feature,
                            points="all",
                            title=f"{feature} vs AHP_class",
                            color='AHP_class',
                            color_discrete_map={'A': 'green', 'B': 'gold', 'C': 'red'},
                            category_orders={'AHP_class': ['A', 'B', 'C']}
                        )

                        # Calcular medias por clase
                        means_ahp = merged.groupby('AHP_class')[feature].mean().round(3)

                        # Mostrar gráfico
                        st.plotly_chart(fig_px, use_container_width=True)

                        # Mostrar medias como leyenda textual
                        ordered_classes = ['A', 'B', 'C']
                        means_text = " ".join([
                            f"<b>Media {cls}:</b> {means_ahp[cls]:,.2f}" 
                            for cls in ordered_classes if cls in means_ahp
                        ])
                        st.markdown(
                            f"<div style='text-align:center; font-size: 0.9rem;'>{means_text}</div>", 
                            unsafe_allow_html=True
                        )

                    except Exception as e:
                        st.warning(f"No se pudo graficar {feature} por AHP_class: {e}")

                # --- ABC_class ---
                with fig_col2:
                    try:
                        # Crear el boxplot base
                        fig_px2 = px.box(
                            merged,
                            x='ABC_class',
                            y=feature,
                            points="all",
                            title=f"{feature} vs ABC_class",
                            color='ABC_class',
                            color_discrete_map={'A': 'green', 'B': 'gold', 'C': 'red'},
                            category_orders={'ABC_class': ['A', 'B', 'C']}
                        )

                        # Calcular medias por clase
                        means_abc = merged.groupby('ABC_class')[feature].mean().round(3)

                        # Mostrar gráfico
                        st.plotly_chart(fig_px2, use_container_width=True)

                        # Mostrar medias como leyenda textual
                        ordered_classes = ['A', 'B', 'C']
                        means_text2 = " ".join([
                            f"<b>Media {cls}:</b> {means_abc[cls]:,.2f}" 
                            for cls in ordered_classes if cls in means_abc
                        ])
                        st.markdown(
                            f"<div style='text-align:center; font-size: 0.9rem;'>{means_text2}</div>", 
                            unsafe_allow_html=True
                        )


                    except Exception as e:
                        st.warning(f"No se pudo graficar {feature} por ABC_class: {e}")




    # Botón de descarga
    st.subheader("Descargar resumen")
    to_download = merged.copy()
    csv = to_download.to_csv(index=False).encode('utf-8')
    st.download_button(label="Descargar CSV", data=csv, file_name=f"{view_mode}_summary.csv", mime='text/csv')

    st.success("Ejecución completada.")

# Fin del botón ejecutar
