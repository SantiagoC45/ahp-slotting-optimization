# app.py
"""
App Streamlit: clasificación ABC vs AHP para Shipping y Labor.
Mantener nombres de pestañas: "Shipping Detail Report" y "Labor Activity Report".
Se añadió sección para subir archivos de Demanda/Forecast y reclasificación.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import json2html
from io import BytesIO
import io
from st_aggrid import AgGrid, GridOptionsBuilder
from ABCenv import WarehouseEnvCalc
import numpy as np

from ahp import (
    compute_abc,
    compute_ahp,
    compute_similarity_metrics,
    compute_summary,
    get_quarter_from_date,
    make_abc_classification,
    process_dataset,
    reclasify_final,
)


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
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df[(df[date_col] >= start) & (df[date_col] <= end)].copy()


st.set_page_config(layout="wide", page_title="ABC vs AHP Dashboard")

st.title("Clasificación ABC vs AHP — Shipping & Labor")
st.markdown(
    "**Instrucciones:** Subir un archivo Excel que contenga *exactamente* las hojas `Shipping Detail Report` y `Labor Activity Report` (no renombres las pestañas)."
)

uploaded = st.file_uploader("Sube archivo Excel (.xlsx) con ambas hojas", type=["xlsx"])

if not uploaded:
    st.info(
        "Sube un archivo Excel para empezar. Asegúrate de que las hojas se llamen exactamente 'Shipping Detail Report' y 'Labor Activity Report'."
    )
    st.stop()

# Cargar archivos (cacheado)
try:
    df_ship_raw, df_labor_raw = load_excel(uploaded)
except Exception as e:
    st.exception(e)
    st.stop()


st.sidebar.header("Configuración general")
view_mode = st.sidebar.selectbox("Ver dataset", options=["Shipping", "Labor"])
# Preview y selección de columna fecha
st.subheader("Preview y selección de columna fecha")

col1, col2 = st.columns(2)
with col1:
    st.write("**Shipping**")
    st.dataframe(df_ship_raw.head())
    ship_date_col = st.selectbox(
        "Selecciona columna fecha (Shipping)",
        options=list(df_ship_raw.columns),
        index=0,
        key="ship_date",
    )
with col2:
    st.write("**Labor**")
    st.dataframe(df_labor_raw.head())
    labor_date_col = st.selectbox(
        "Selecciona columna fecha (Labor)",
        options=list(df_labor_raw.columns),
        index=0,
        key="labor_date",
    )

# Convertir a datetime solo una vez
df_ship_raw[ship_date_col] = pd.to_datetime(df_ship_raw[ship_date_col], errors="coerce")
df_labor_raw[labor_date_col] = pd.to_datetime(
    df_labor_raw[labor_date_col], errors="coerce"
)

# Calcular fechas mín/máx
min_date = min(df_ship_raw[ship_date_col].min(), df_labor_raw[labor_date_col].min())
max_date = max(df_ship_raw[ship_date_col].max(), df_labor_raw[labor_date_col].max())

st.sidebar.subheader("Filtrar rango de fechas (aplicado a ambos datasets)")
date_range = st.sidebar.date_input(
    "Rango fechas",
    value=[min_date.date(), max_date.date()],
    min_value=min_date.date(),
    max_value=max_date.date(),
)
start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# Aplicar filtro (cacheado)
df_ship = preprocess_data(df_ship_raw, ship_date_col, start, end)
df_labor = preprocess_data(df_labor_raw, labor_date_col, start, end)


st.sidebar.write(
    f"Shipping filtrado: {df_ship.shape[0]} filas \nLabor filtrado: {df_labor.shape[0]} filas"
)

# ---------------------------
# Parámetros ABC (umbral editable)
# ---------------------------
st.sidebar.subheader("Umbrales ABC (cortes acumulativos en %)")
default_cuts = [80, 90]
cut_a = st.sidebar.slider(
    "A hasta (%)", min_value=1, max_value=99, value=default_cuts[0]
)
cut_b = st.sidebar.slider(
    "B hasta (%)", min_value=cut_a + 1, max_value=100, value=default_cuts[1]
)
cuts = [cut_a, cut_b]

# ---------------------------
# Selección columnas SKU y agregación
# ---------------------------
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
if view_mode == "Shipping":
    df_use = df_ship
    candidate_features_ship = [qty_col_ship, weight_col_ship, boxes_col_ship]
    default_features = [f for f in candidate_features_ship if f in df_use.columns]
else:
    df_use = df_labor
    candidate_features_labor = [weight_col_labor, qty_col_labor]
    default_features = [f for f in candidate_features_labor if f in df_use.columns]

# Detectar columnas numéricas
numeric_cols = df_use.select_dtypes(include=["int64", "float64"]).columns.tolist()

st.sidebar.markdown(f"### ⚙️ Selección de variables para AHP ({view_mode})")

if not numeric_cols:
    st.sidebar.warning(
        f"No se encontraron columnas numéricas en el dataset de {view_mode}."
    )
    use_features = []
else:
    use_features = st.sidebar.multiselect(
        f"Selecciona variables numéricas para AHP ({view_mode})",
        options=numeric_cols,
        default=default_features if default_features else numeric_cols,
    )

# ---------------------------
# Panel AHP: subir imagen explicativa y editar matriz
# ---------------------------
st.header("Panel AHP")
st.markdown("### Escala de comparación por pares (Método AHP)")

st.markdown(
    """
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
"""
)

st.info(
    "💡 Usa esta tabla como guía para llenar la **matriz de comparaciones** o definir los **pesos directos** de los criterios."
)

# Crear interfaz para editar matriz de comparaciones entre criterios (features)
st.subheader("Editar matriz de comparaciones (AHP)")
st.markdown(
    "Usa valores 1,3,5,7,9. Puedes editar cada par. Alternativa: editar pesos directos."
)

# Preparar estructura de comparaciones para ahpy: dict of dicts
features = use_features.copy()
if len(features) < 2:
    st.warning("Selecciona al menos 2 features para ejecutar AHP.")
    st.stop()

pairs = []
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        pairs.append({"feature_i": features[i], "feature_j": features[j], "value": 1.0})

pairs_df = pd.DataFrame(pairs)
edited_pairs = st.data_editor(pairs_df, num_rows="dynamic", key="ahp_pairs_editor")

# Convertir a comparisons dict para ahpy
comparisons = {}
for _, row in edited_pairs.iterrows():
    i = row["feature_i"]
    j = row["feature_j"]
    val = float(row["value"]) if pd.notna(row["value"]) else 1.0
    comparisons[(i, j)] = val

# Opción alternativa: editar pesos directos
st.markdown("**Opción:** editar pesos directos (sobreescribe la matriz si se usan).")

weights_manual = {}
use_manual_weights = st.checkbox(
    "Usar pesos manuales (si activo, la matriz se ignorará para pesos)"
)

if use_manual_weights:
    col1, col2 = st.columns(2)
    for f in features:
        weights_manual[f] = col1.number_input(
            f"Peso {f}", min_value=0.0, value=0.0, step=0.05, key=f"w_{f}"
        )
    total_w = sum(weights_manual.values())
    st.markdown(f"**Suma actual de pesos:** {total_w:.3f}")
    progress_value = min(total_w, 1.0)
    st.progress(progress_value)
    if abs(total_w - 1.0) > 1e-6:
        st.warning(
            f"La suma de los pesos ({total_w:.2f}) debe ser exactamente 1. Ajusta los valores antes de continuar."
        )
        st.stop()
    else:
        for k in weights_manual:
            weights_manual[k] = weights_manual[k] / total_w
        st.success("✅ Pesos válidos: la suma es 1.")

# ---------------------------
# NUEVA SECCIÓN: Upload de archivos de Demand / Forecast
# ---------------------------
if view_mode == "Labor":
    st.sidebar.subheader("Demand / Forecast")
    st.sidebar.markdown(
        "Puedes subir uno o varios archivos de demanda/forecast (Excel o CSV). El flujo detecta si el archivo tiene columna fecha o no y procesa por trimestre si aplica."
    )
    demand_files = st.sidebar.file_uploader(
        "Sube archivos Demand/Forecast (xlsx/csv)",
        accept_multiple_files=True,
        type=["xlsx", "xls", "csv"],
    )
else:
    demand_files = None

# Botón ejecutar
# --- BUTTON: Ejecutar clasificación y métricas ---
if st.button("Ejecutar clasificación y métricas"):
    # --- Preparar df_use y sku col según modo ---
    if view_mode == "Shipping":
        df_use = df_ship.copy()
        sku_col = sku_col_ship
    else:
        df_use = df_labor.copy()
        sku_col = sku_col_labor

    # Construir aggregations basadas en features seleccionadas
    if use_features and len(use_features) > 0:
        aggregations = {col: col for col in use_features}
    else:
        if view_mode == "Shipping":
            aggregations = {
                "Qty Shipped": qty_col_ship,
                "Weight [Kg]": weight_col_ship,
                "Boxes": boxes_col_ship,
            }
        else:
            aggregations = {"Weight [Kg]": weight_col_labor, "Pick Unit": qty_col_labor}

    # Calcular summary por SKU
    summary = compute_summary(
        df_use, mode=view_mode, sku_col=sku_col, aggregations=aggregations
    )

    # ABC clásico
    qty_col = "Qty Shipped" if view_mode == "Shipping" else "Pick Unit"
    abc_df = compute_abc(summary, qty_col=qty_col, cuts=cuts)

    # AHP
    if use_manual_weights:
        ahp_summary, criteria = compute_ahp(
            summary,
            features=features,
            comparisons_dict=comparisons,
            cuts=cuts,
            w=weights_manual,
        )
    else:
        ahp_summary, criteria = compute_ahp(
            summary, features=features, comparisons_dict=comparisons, cuts=cuts
        )

    # Merge principal
    abc_cols = ["SKU"] + use_features + ["cum%", "ABC_class"]
    abc_cols = [c for c in abc_cols if c in abc_df.columns]
    ahp_cols = ["SKU", "AHP_score", "cum_AHP%", "AHP_class"] + [
        f"{f}_norm" for f in features if f"{f}_norm" in ahp_summary.columns
    ]
    merged = pd.merge(
        abc_df[abc_cols], ahp_summary[ahp_cols], on="SKU", how="outer"
    ).fillna(0)

    # Guardar en sesión
    st.session_state["merged"] = merged
    st.session_state["criteria"] = criteria
    st.session_state["features"] = features

# --- Mostrar resultados si existen ---
if "merged" in st.session_state:
    merged = st.session_state["merged"]
    criteria = st.session_state.get("criteria", None)
    features = st.session_state.get("features", [])

    # Mostrar resultados comparativos
    st.header("Resultados comparativos")
    st.subheader("Tabla de resumen (por SKU)")
    st.dataframe(merged)

    # Conteos por clase
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("ABC - A count", int((merged["ABC_class"] == "A").sum()))
        st.metric("AHP - A count", int((merged["AHP_class"] == "A").sum()))
    with colB:
        st.metric("ABC - B count", int((merged["ABC_class"] == "B").sum()))
        st.metric("AHP - B count", int((merged["AHP_class"] == "B").sum()))
    with colC:
        st.metric("ABC - C count", int((merged["ABC_class"] == "C").sum()))
        st.metric("AHP - C count", int((merged["AHP_class"] == "C").sum()))

    # Métricas de similitud
    st.subheader("Métricas de similitud por clase y globales")
    merged_for_metrics = merged.copy()
    metrics_df, global_indices = compute_similarity_metrics(
        merged_for_metrics,
        features=[f"{f}_norm" for f in features],
        class_col="AHP_class",
    )
    st.dataframe(metrics_df)
    st.write("Índices globales:", global_indices)

    # --- BOXLOTS COMPARATIVOS ---
    st.subheader("Boxplots comparativos")

    if not features:
        st.warning("Por favor selecciona al menos una feature para graficar.")
    else:
        plot_features = [f for f in features if f in merged.columns]

        if not plot_features:
            st.warning(
                "Ninguna de las features seleccionadas existe en los datos combinados."
            )
        else:
            st.markdown("### Plotly interactivos")
            for feature in plot_features:
                fig_col1, fig_col2 = st.columns(2)

                # --- AHP_class ---
                with fig_col1:
                    try:
                        fig_px = px.box(
                            merged,
                            x="AHP_class",
                            y=feature,
                            points="all",
                            title=f"{feature} vs AHP_class",
                            color="AHP_class",
                            color_discrete_map={"A": "green", "B": "gold", "C": "red"},
                            category_orders={"AHP_class": ["A", "B", "C"]},
                        )
                        means_ahp = merged.groupby("AHP_class")[feature].mean().round(3)
                        st.plotly_chart(fig_px, use_container_width=True)
                        ordered_classes = ["A", "B", "C"]
                        means_text = " ".join(
                            [
                                f"<b>Media {cls}:</b> {means_ahp[cls]:,.2f}"
                                for cls in ordered_classes
                                if cls in means_ahp
                            ]
                        )
                        st.markdown(
                            f"<div style='text-align:center; font-size: 0.9rem;'>{means_text}</div>",
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.warning(f"No se pudo graficar {feature} por AHP_class: {e}")

                # --- ABC_class ---
                with fig_col2:
                    try:
                        fig_px2 = px.box(
                            merged,
                            x="ABC_class",
                            y=feature,
                            points="all",
                            title=f"{feature} vs ABC_class",
                            color="ABC_class",
                            color_discrete_map={"A": "green", "B": "gold", "C": "red"},
                            category_orders={"ABC_class": ["A", "B", "C"]},
                        )
                        means_abc = merged.groupby("ABC_class")[feature].mean().round(3)
                        st.plotly_chart(fig_px2, use_container_width=True)
                        ordered_classes = ["A", "B", "C"]
                        means_text2 = " ".join(
                            [
                                f"<b>Media {cls}:</b> {means_abc[cls]:,.2f}"
                                for cls in ordered_classes
                                if cls in means_abc
                            ]
                        )
                        st.markdown(
                            f"<div style='text-align:center; font-size: 0.9rem;'>{means_text2}</div>",
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.warning(f"No se pudo graficar {feature} por ABC_class: {e}")

    # --- SIMULACIÓN (persiste abajo) ---
    st.header("Calcule el puntaje de la clasificación con una simulación")
    st.markdown("Selecciona la fecha inicial y la cantidad de días para la simulación.")

    col1_sim, col2_sim = st.columns(2)
    with col1_sim:
        default_date = (
            start.date() if "start" in locals() else pd.Timestamp.now().date()
        )
        sim_init_date = st.date_input(
            "Fecha inicial de simulación", value=default_date, key="sim_init_date"
        )
    with col2_sim:
        sim_days = st.number_input(
            "Días a simular",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            key="sim_days",
        )

    if st.button("Simular"):
        try:
            env = WarehouseEnvCalc(int(sim_days), np.datetime64(sim_init_date))
            df_for_sim = merged.copy()
            if "SKU" in df_for_sim.columns:
                df_for_sim = df_for_sim.set_index("SKU")
            score_abc = env.CalcDataframeScore(df_for_sim, "ABC_class")
            score_ahp = env.CalcDataframeScore(df_for_sim, "AHP_class")
            st.session_state["sim_results"] = (score_abc, score_ahp)
        except Exception as e:
            st.exception(e)

    if "sim_results" in st.session_state:
        score_abc, score_ahp = st.session_state["sim_results"]
        st.subheader("Resultados de la simulación")
        st.write(f"Puntaje usando ABC_class: {score_abc}")
        st.write(f"Puntaje usando AHP_class: {score_ahp}")

    # ---------------------------
    # NUEVO: Procesamiento de archivos Demand / Forecast (si hay)
    # ---------------------------
    final_demand_df = None

    if demand_files and view_mode == "Labor":
        st.header("Reclasificación por Demanda / Forecast")
        st.markdown(
            "Se procesarán los archivos subidos. Si el archivo contiene fechas, se separa por trimestres."
        )

        st.markdown(
            """
        ### - Criterio de reclasificación

        **¿Cómo se determina la clase final de cada SKU?**

        - **Si el SKU no existe en la tabla de Labor (AHP):**
            - Se asigna la clasificación de **ABC**.
                    
        - **Si el SKU sí existe en Labor (AHP):**
            - Si la clase **LABOR_AHP** y la clase **ABC** coinciden, se mantiene la de **LABOR**.
            - Si difieren, se escoge la categoría con mayor prioridad:<br>
                <b>Prioridad:</b> <span style='color:#1976d2'><b>A</b></span> &gt; <span style='color:#fbc02d'><b>B</b></span> &gt; <span style='color:#d32f2f'><b>C</b></span>

        """,
            unsafe_allow_html=True,
        )

        # Preparar labor_small (solo SKU, LABOR_AHP_class, LABOR_cum_pct)
        labor_small = ahp_labor_small = None
        try:
            labor_small = ahp_summary = None
            if (
                "AHP_class" in ahp_summary.columns
                if "ahp_summary" in locals()
                else False
            ):
                pass
        except Exception:
            pass

        # Construir labor_small desde ahp_summary si existe
        if "ahp_summary" in locals() and isinstance(ahp_summary, pd.DataFrame):
            labor_small = ahp_summary[["SKU", "AHP_class", "cum_AHP%"]].copy()
            labor_small.rename(
                columns={"AHP_class": "LABOR_AHP_class", "cum_AHP%": "LABOR_cum_pct"},
                inplace=True,
            )
        else:
            # Si por alguna razón no existe ahp_summary, intentamos obtenerlo desde 'merged'
            if "merged" in locals():
                # merged puede tener AHP_class y cum_AHP%
                if "AHP_class" in merged.columns:
                    labor_small = merged[["SKU", "AHP_class", "cum_AHP%"]].copy()
                    labor_small.rename(
                        columns={
                            "AHP_class": "LABOR_AHP_class",
                            "cum_AHP%": "LABOR_cum_pct",
                        },
                        inplace=True,
                    )
        if labor_small is None or labor_small.empty:
            st.warning(
                "No se pudo construir labor_small (información AHP de labor). La reclasificación por demanda requiere que AHP para Labor se haya calculado correctamente."
            )
        else:
            all_results = []
            for f in demand_files:
                # Leer archivo (xlsx o csv)
                try:
                    if f.name.lower().endswith((".xls", ".xlsx")):
                        # intentar leer primeras hojas relevantes; si varias hojas, leer la primera
                        try:
                            xls = pd.ExcelFile(f)
                            # Si hay hojas con 'Demand' o 'Forecast' en el nombre, priorizar
                            sheet_to_use = None
                            for s in xls.sheet_names:
                                if any(x in s.lower() for x in ["demand", "forecast"]):
                                    sheet_to_use = s
                                    break
                            if sheet_to_use is None:
                                sheet_to_use = xls.sheet_names[0]
                            df_d = pd.read_excel(xls, sheet_name=sheet_to_use)
                        except Exception as e:
                            st.warning(f"No se pudo leer {f.name} como Excel: {e}")
                            continue
                    else:
                        # csv
                        try:
                            df_d = pd.read_csv(f)
                        except Exception as e:
                            st.warning(f"No se pudo leer {f.name} como CSV: {e}")
                            continue

                    # Normalizar SKU y columnas
                    if "SKU" in df_d.columns:
                        df_d["SKU"] = df_d["SKU"].astype(str).str.upper()

                    # Detectar dataset name/label
                    dataset_label = getattr(f, "name", "UploadedDataset")

                    # Procesar dataset con process_dataset (usa get_quarter_from_date internamente)
                    try:
                        res = process_dataset(
                            df_d, labor_small, dataset_name=dataset_label
                        )
                        if isinstance(res, pd.DataFrame) and not res.empty:
                            all_results.append(res)
                        else:
                            st.info(f"No se generó output para {dataset_label}.")
                    except Exception as e:
                        st.warning(f"Error procesando {dataset_label}: {e}")
                except Exception as e:
                    st.warning(f"Error leyendo archivo {f.name}: {e}")

            if all_results:
                final_demand_df = pd.concat(all_results, ignore_index=True)
                st.subheader(
                    "Resumen combinado de reclasificaciones (todos los archivos / trimestres)"
                )

                st.dataframe(final_demand_df)

                # Estadística de cambio
                total_skus = len(final_demand_df)
                n_changed = int(final_demand_df["Changed"].sum())
                pct_changed = (
                    100 * final_demand_df["Changed"].mean() if total_skus > 0 else 0.0
                )

                st.metric("Total SKUs procesados", total_skus)
                st.metric(
                    "SKUs cuyo class cambió vs LABOR",
                    f"{n_changed} ({pct_changed:.2f}%)",
                )

                # Pie chart de cambiado / no cambiado
                st.subheader("Proporción de SKUs que cambiaron su clase por Quarter")

                quarters = sorted(final_demand_df["Quarter"].unique())
                n = len(quarters)

                if n == 1:
                    # Solo uno, normal
                    df_q = final_demand_df[final_demand_df["Quarter"] == quarters[0]]
                    pie_q = px.pie(
                        df_q.groupby("Changed").size().reset_index(name="count"),
                        names=df_q.groupby("Changed")
                        .size()
                        .reset_index(name="count")["Changed"]
                        .map({True: "Changed", False: "Unchanged"}),
                        values="count",
                        title=f"Proporción de SKUs que cambiaron su clase en {quarters[0]}",
                    )
                    st.plotly_chart(pie_q, use_container_width=True)

                elif n == 2:
                    # Dos, lado a lado
                    cols = st.columns(2)
                    for i in range(2):
                        with cols[i]:
                            df_q = final_demand_df[
                                final_demand_df["Quarter"] == quarters[i]
                            ]
                            pie_q = px.pie(
                                df_q.groupby("Changed")
                                .size()
                                .reset_index(name="count"),
                                names=df_q.groupby("Changed")
                                .size()
                                .reset_index(name="count")["Changed"]
                                .map({True: "Changed", False: "Unchanged"}),
                                values="count",
                                title=f"Proporción de SKUs que cambiaron su clase en {quarters[i]}",
                            )
                            st.plotly_chart(pie_q, use_container_width=True)

                elif n == 3:
                    # Dos arriba, uno abajo centrado
                    cols_top = st.columns(2)
                    for i in range(2):
                        with cols_top[i]:
                            df_q = final_demand_df[
                                final_demand_df["Quarter"] == quarters[i]
                            ]
                            pie_q = px.pie(
                                df_q.groupby("Changed")
                                .size()
                                .reset_index(name="count"),
                                names=df_q.groupby("Changed")
                                .size()
                                .reset_index(name="count")["Changed"]
                                .map({True: "Changed", False: "Unchanged"}),
                                values="count",
                                title=f"Proporción de SKUs que cambiaron su clase en {quarters[i]}",
                            )
                            st.plotly_chart(pie_q, use_container_width=True)
                    # Uno abajo centrado
                    st.write("")  # Espacio
                    col_center = st.columns([0.25, 0.5, 0.25])
                    with col_center[1]:
                        df_q = final_demand_df[
                            final_demand_df["Quarter"] == quarters[2]
                        ]
                        pie_q = px.pie(
                            df_q.groupby("Changed").size().reset_index(name="count"),
                            names=df_q.groupby("Changed")
                            .size()
                            .reset_index(name="count")["Changed"]
                            .map({True: "Changed", False: "Unchanged"}),
                            values="count",
                            title=f"Proporción de SKUs que cambiaron su clase en {quarters[2]}",
                        )
                        st.plotly_chart(pie_q, use_container_width=True)

                elif n == 4:
                    # Dos arriba, dos abajo
                    cols_top = st.columns(2)
                    for i in range(2):
                        with cols_top[i]:
                            df_q = final_demand_df[
                                final_demand_df["Quarter"] == quarters[i]
                            ]
                            pie_q = px.pie(
                                df_q.groupby("Changed")
                                .size()
                                .reset_index(name="count"),
                                names=df_q.groupby("Changed")
                                .size()
                                .reset_index(name="count")["Changed"]
                                .map({True: "Changed", False: "Unchanged"}),
                                values="count",
                                title=f"Proporción de SKUs que cambiaron su clase en {quarters[i]}",
                            )
                            st.plotly_chart(pie_q, use_container_width=True)
                    cols_bottom = st.columns(2)
                    for i in range(2, 4):
                        with cols_bottom[i - 2]:
                            df_q = final_demand_df[
                                final_demand_df["Quarter"] == quarters[i]
                            ]
                            pie_q = px.pie(
                                df_q.groupby("Changed")
                                .size()
                                .reset_index(name="count"),
                                names=df_q.groupby("Changed")
                                .size()
                                .reset_index(name="count")["Changed"]
                                .map({True: "Changed", False: "Unchanged"}),
                                values="count",
                                title=f"Proporción de SKUs que cambiaron su clase en {quarters[i]}",
                            )
                            st.plotly_chart(pie_q, use_container_width=True)

                else:
                    # Más de 4: en filas de a 2
                    for i in range(0, n, 2):
                        cols = st.columns(2)
                        for j in range(2):
                            if i + j < n:
                                with cols[j]:
                                    df_q = final_demand_df[
                                        final_demand_df["Quarter"] == quarters[i + j]
                                    ]
                                    pie_q = px.pie(
                                        df_q.groupby("Changed")
                                        .size()
                                        .reset_index(name="count"),
                                        names=df_q.groupby("Changed")
                                        .size()
                                        .reset_index(name="count")["Changed"]
                                        .map({True: "Changed", False: "Unchanged"}),
                                        values="count",
                                        title=f"Proporción de SKUs que cambiaron su clase en {quarters[i + j]}",
                                    )
                                    st.plotly_chart(pie_q, use_container_width=True)

                # Bar chart: count por Source (LABOR/ABC) y por Quarter
                bar = (
                    final_demand_df.groupby(["Quarter", "Source"])
                    .size()
                    .reset_index(name="count")
                )
                fig_bar = px.bar(
                    bar,
                    x="Quarter",
                    y="count",
                    color="Source",
                    barmode="group",
                    title="Conteo por Quarter y Source (LABOR/ABC)",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Tabla de detalle (top cambios)
                st.subheader("Detalle de SKUs que cambiaron (top 50)")
                changed_df = final_demand_df[final_demand_df["Changed"]].sort_values(
                    "Used_Qty", ascending=False
                )
                st.dataframe(changed_df.head(50))

            else:
                st.info("No se generaron resultados a partir de los archivos subidos.")

        if final_demand_df is not None:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                final_demand_df.to_excel(
                    writer, index=False, sheet_name="Reclasificacion"
                )
            st.download_button(
                label="Descargar resultado reclasificación (Excel)",
                data=output.getvalue(),
                file_name="reclasificacion_demand.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # Descargar merged principal
    output_main = io.BytesIO()
    with pd.ExcelWriter(output_main, engine="xlsxwriter") as writer:
        merged.to_excel(writer, index=False, sheet_name="ResumenPrincipal")
    st.download_button(
        label="Descargar resumen principal (Excel)",
        data=output_main.getvalue(),
        file_name=f"{view_mode}_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.success("Ejecución completada.")
