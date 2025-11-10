import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import ahpy

def compute_summary(df, sku_col='SKU', mode='Shipping', aggregations=None):
    """
    Agrupa por SKU y devuelve un resumen:
      - Si mode='shipping': Qty Shipped, Weight [Kg], Boxes
      - Si mode='labor': Weight [Kg], Pick Unit
    Si las columnas no existen, las crea con 0 para evitar errores.
    """
    # Definir columnas por defecto según modo
    defaults = {
        'Shipping': {'Qty Shipped': 'Qty Shipped', 'Weight [Kg]': 'Weight [Kg]', 'Boxes': 'Boxes'},
        'Labor': {'Weight [Kg]': 'Weight [Kg]', 'Pick Unit': 'Pick Unit'}
    }

    # Determinar agregaciones base
    if aggregations is None:
        aggregations = defaults.get(mode.lower(), defaults[mode])

    # Asegurar existencia de columnas
    for col in aggregations.values():
        if col not in df.columns:
            df[col] = 0

    # Preparar diccionario para agregación
    agg_dict = {v: 'sum' for v in aggregations.values()}

    # Agrupar y renombrar
    summary = (
        df.groupby(sku_col)
          .agg(agg_dict)
          .reset_index()
          .rename(columns={v: k for k, v in aggregations.items()})
    )

    # Evitar NaNs
    for col in aggregations.keys():
        summary[col] = summary[col].fillna(0)

    return summary


def compute_abc(summary, qty_col, cuts=[80,90]):
    """
    Clasificación ABC clásica por Pareto basada en `qty_col`.
    - cuts: lista con percentiles acumulativos (ej [80,90]) => A: <=80, B: >80-90, C: >90-100
    Devuelve summary con 'cum%', 'ABC_class'
    """
    df = summary.copy()
    # Orden descendente por qty
    df = df.sort_values(qty_col, ascending=False).reset_index(drop=True)
    total = df[qty_col].sum()
    # Si total 0, evitar división por cero; asignar 0%
    if total == 0:
        df['cum%'] = 0.0
    else:
        df['cum%'] = (df[qty_col].cumsum() / total) * 100
    # Definir cortes
    cuts_sorted = sorted(cuts)
    def classify(x):
        if x <= cuts_sorted[0]:
            return 'A'
        elif x <= cuts_sorted[1]:
            return 'B'
        else:
            return 'C'
    df['ABC_class'] = df['cum%'].apply(classify)
    return df

def compute_ahp(summary, features, comparisons_dict, cuts=[80,90], w=None):
    """
    Ejecuta AHP con ahpy usando comparisons_dict (pares de comparaciones) y calcula AHP_score.
    - features: lista de columnas sobre las que construir el score (columnas numéricas de summary)
    - comparisons_dict: diccionario con comparaciones en el formato (compatible con ahpy.Compare)
    Devuelve summary con columnas normalizadas, 'AHP_score', 'cum_AHP%', 'AHP_class' y retorna también el objeto criteria de ahpy.
    """
    df = summary.copy().reset_index(drop=True)
    # Normalizar columnas (col / col.max())
    for col in features:
        if col not in df.columns:
            df[col] = 0.0
        maxv = df[col].max()
        if maxv == 0:
            df[f'{col}_norm'] = 0.0
        else:
            df[f'{col}_norm'] = df[col] / maxv

    # Ejecutar ahpy: Construir objeto Compare
    #try:
    if w:
        criteria = None
        weights = w
    else:  
        criteria = ahpy.Compare('criteria', comparisons=comparisons_dict, precision=6)
        weights = criteria.target_weights  # dict {feature: weight}

    # Asegurar que todos features tengan un peso (si el usuario escribió pesos directos, manejar)
    weights_list = [weights.get(f, 0) for f in features]
    #except Exception as e:
        # Si ahpy falla, devolver pesos iguales
        # st.warning("ahpy no pudo construir la matriz AHP con las comparaciones dadas. Se usarán pesos iguales.")
        # weights_list = [1.0/len(features)]*len(features)
        # criteria = None

    # Calcular AHP_score = suma(weights[i] * feature_norm)
    # Si criteria existe, usar sus pesos; si no, usar weights_list
    norm_cols = [f'{col}_norm' for col in features]
    w = np.array(weights_list, dtype=float)
    norms = df[norm_cols].fillna(0).values
    # Normalizar pesos para que sumen 1 si no están normalizados
    if w.sum() == 0:
        w = np.ones(len(features))/len(features)
    else:
        w = w / w.sum()
    df['AHP_score'] = norms.dot(w)
    # cumsum por puntuación descendente
    df = df.sort_values('AHP_score', ascending=False).reset_index(drop=True)
    total_ahp = df['AHP_score'].sum()
    if total_ahp == 0:
        df['cum_AHP%'] = 0.0
    else:
        df['cum_AHP%'] = (df['AHP_score'].cumsum() / total_ahp) * 100
    # Clasificación A/B/C con mismos cortes
    cuts_sorted = sorted(cuts)
    def classify_ahp(x):
        if x <= cuts_sorted[0]:
            return 'A'
        elif x <= cuts_sorted[1]:
            return 'B'
        else:
            return 'C'
    df['AHP_class'] = df['cum_AHP%'].apply(classify_ahp)

    return df, criteria

def compute_similarity_metrics(summary, features, class_col='AHP_class'):
    """
    Calcula métricas por clase:
    - mean & median cosine similarity (pairwise entre elementos dentro de cada clase)
    - mean euclidean distance (pairwise)
    - coeficiente de variación (std/mean) del AHP_score por clase
    También calcula índices globales: silhouette, Davies-Bouldin, Calinski-Harabasz (si aplicable).
    Devuelve: (df_metrics_por_clase, dict_global_indices)
    """
    df = summary.copy().reset_index(drop=True)
    results = []
    X = df[features].fillna(0).values
    labels = df[class_col].values
    # Por clase
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        sub = X[idx]
        res = {'class': cls, 'n': len(idx)}
        if len(idx) < 2:
            res.update({
                'mean_cosine_sim': np.nan,
                'median_cosine_sim': np.nan,
                'mean_euclidean': np.nan,
                'cv_ahp_score': np.nan
            })
        else:
            # cosine similarity matriz (pairwise), tomar upper triangle sin diagonal
            cos_mat = cosine_similarity(sub)
            iu = np.triu_indices_from(cos_mat, k=1)
            cos_vals = cos_mat[iu]
            res['mean_cosine_sim'] = np.mean(cos_vals) if cos_vals.size>0 else np.nan
            res['median_cosine_sim'] = np.median(cos_vals) if cos_vals.size>0 else np.nan
            # euclidian distances
            dists = pairwise_distances(sub, metric='euclidean')
            dists_vals = dists[iu]
            res['mean_euclidean'] = np.mean(dists_vals) if dists_vals.size>0 else np.nan
            # Coef de variación del AHP_score en el grupo
            ahp_vals = df.loc[idx, 'AHP_score'].values
            if ahp_vals.mean() == 0:
                res['cv_ahp_score'] = np.nan
            else:
                res['cv_ahp_score'] = ahp_vals.std(ddof=0)/ahp_vals.mean()
        results.append(res)
    df_metrics = pd.DataFrame(results)

    # Global indices: necesitan al menos 2 clusters y cada cluster con >=2 elementos para silhouette
    global_idx = {}
    try:
        # silhouette requires >1 label and n_samples > n_labels
        if len(np.unique(labels)) > 1 and X.shape[0] > len(np.unique(labels)):
            # silhouette_score requiere al menos 2 elementos por cluster? Actually only total > n_clusters.
            global_idx['silhouette'] = silhouette_score(X, labels)
        else:
            global_idx['silhouette'] = np.nan
    except Exception:
        global_idx['silhouette'] = np.nan
    try:
        if len(np.unique(labels)) > 1:
            global_idx['davies_bouldin'] = davies_bouldin_score(X, labels)
            global_idx['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        else:
            global_idx['davies_bouldin'] = np.nan
            global_idx['calinski_harabasz'] = np.nan
    except Exception:
        global_idx['davies_bouldin'] = np.nan
        global_idx['calinski_harabasz'] = np.nan

    return df_metrics, global_idx


# -----------------------------
# NUEVAS FUNCIONES: manejo de forecast/demand y reclasificación
# -----------------------------

def get_quarter_from_date(date):
    """Devuelve Q1..Q4 según el mes de una fecha (acepta pandas Timestamp)."""
    if pd.isna(date):
        return None
    m = date.month
    if 1 <= m <= 3:
        return "Q1"
    elif 4 <= m <= 6:
        return "Q2"
    elif 7 <= m <= 9:
        return "Q3"
    else:
        return "Q4"


def make_abc_classification(df, sku_col="SKU", qty_cols=("Qty", "Delivered Qty")):
    """
    Crea una clasificación ABC basada en la primera columna de cantidad que encuentre en qty_cols.
    Devuelve DataFrame con columnas: SKU, Used_Qty, cum_qty, cum_pct, ABC_class
    """
    temp = df.copy()
    # detectar columna qty válida
    qty_col = None
    for col in qty_cols:
        if col in temp.columns:
            qty_col = col
            break
    if qty_col is None:
        raise ValueError("No se encontró 'Qty' ni 'Delivered Qty' en el dataset.")

    # Normalizar SKU a str y mayúsculas si existe
    if sku_col in temp.columns:
        temp[sku_col] = temp[sku_col].astype(str).str.upper()
    else:
        raise ValueError(f"No se encontró la columna SKU ('{sku_col}') en el dataset.")

    # Agrupar por SKU sumando la cantidad detectada
    temp = temp.groupby(sku_col)[qty_col].sum().reset_index()
    temp = temp.sort_values(qty_col, ascending=False).reset_index(drop=True)
    temp = temp.rename(columns={qty_col: "Used_Qty"})
    temp["cum_qty"] = temp["Used_Qty"].cumsum()
    total = temp["Used_Qty"].sum()
    if total == 0:
        temp["cum_pct"] = 0.0
    else:
        temp["cum_pct"] = 100 * temp["cum_qty"] / total

    temp["ABC_class"] = pd.cut(
        temp["cum_pct"],
        bins=[-0.001, 80, 90, 100],
        labels=["A", "B", "C"],
        include_lowest=True
    ).astype(str)

    return temp


def reclasify_final(row):
    """
    Dado un row con columnas 'LABOR_AHP_class' y 'ABC_class' devuelve (Final_class, Source).
    Source indica si la clase final viene de 'LABOR' o 'ABC'.
    """
    labor_class = row.get("LABOR_AHP_class", None)
    abc_class = row.get("ABC_class", None)
    # Si no hay ABC (improbable), devolver LABOR si existe
    if pd.isna(abc_class) or abc_class is None:
        return labor_class, "LABOR" if not pd.isna(labor_class) else None

    # Si no hay LABOR -> usar ABC
    if pd.isna(labor_class) or labor_class is None:
        return abc_class, "ABC"

    # Si coinciden -> mantener LABOR
    if labor_class == abc_class:
        return labor_class, "LABOR"

    # Si difieren -> categoría con mayor prioridad (A > B > C)
    priority = {"A": 3, "B": 2, "C": 1}
    # seguridad: si hay valores extraños, fallback a ABC
    if labor_class not in priority or abc_class not in priority:
        return abc_class, "ABC"
    final_class = labor_class if priority[labor_class] >= priority[abc_class] else abc_class
    source = "LABOR" if final_class == labor_class else "ABC"
    return final_class, source


def process_single_quarter(df_q, labor_small, quarter_name="Qx"):
    """
    Procesa un dataframe representando un trimestre/forecast sin mezclar con otros,
    aplica ABC por SKU, merge con labor_small (SKU, LABOR_AHP_class, LABOR_cum_pct),
    aplica reclasificación y devuelve df con columnas relevantes.
    """
    # 1. ABC
    abc_df = make_abc_classification(df_q)

    # 2. Merge con LABOR (labor_small: SKU, LABOR_AHP_class, LABOR_cum_pct)
    merged = pd.merge(
        abc_df,
        labor_small,
        on="SKU",
        how="left"
    )

    # 3. Reclasificación (devuelve dos columnas)
    results = merged.apply(lambda r: reclasify_final(r), axis=1, result_type="expand")
    merged["Final_class"] = results[0]
    merged["Source"] = results[1]

    # 4. Cambio frente a LABOR
    merged["Changed"] = merged["Final_class"] != merged["LABOR_AHP_class"]

    # 5. Nombre de Quarter
    merged["Quarter"] = quarter_name

    n = len(merged)
    pct_change = 100 * merged["Changed"].mean() if n > 0 else 0.0

    # Añadir conteos rápidos
    merged["Used_Qty"] = merged["Used_Qty"].fillna(0)

    return merged


def process_dataset(df, labor_small, dataset_name="Dataset"):
    """
    Procesa un dataset de demanda/forecast que puede:
      - no tener columna de fecha (entonces detecta Q en el nombre del dataset)
      - o tener columna de fecha (separa por trimestres usando get_quarter_from_date)
    Devuelve concatenación de resultados por quarter.
    """
    # 1. Detectar columna fecha
    date_cols = ["Inventory Date", "Date", "date", "Fecha", "inventory_date"]
    date_col = None
    for col in date_cols:
        if col in df.columns:
            date_col = col
            break

    # Normalize SKU column name
    if "SKU" in df.columns:
        df["SKU"] = df["SKU"].astype(str).str.upper()

    # CASO A: no hay fecha -> intentar detectar Q por dataset_name
    if date_col is None:
        name_lower = dataset_name.lower()
        detected_q = None
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            if q.lower() in name_lower:
                detected_q = q
                break
        quarter_label = detected_q if detected_q else dataset_name
        result = process_single_quarter(df, labor_small, quarter_label)
        return result

    # CASO B: tiene fechas -> separar por trimestre
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df["Quarter_auto"] = df[date_col].apply(get_quarter_from_date)
    all_results = []
    for q in sorted(df["Quarter_auto"].dropna().unique()):
        df_q = df[df["Quarter_auto"] == q]
        base_name = dataset_name.split()[0] if dataset_name else "Demand"
        q_name = f"{base_name} {q}"  
        result_q = process_single_quarter(df_q, labor_small, q_name)
        all_results.append(result_q)
    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)

