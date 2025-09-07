# app/core.py
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from networkx import DiGraph, draw_networkx
from pgmpy.models import DiscreteBayesianNetwork  
from pgmpy.estimators import PC, ParameterEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from causalinference import CausalModel


# ===========================================================
# 1. Generación de datos observables
# ===========================================================
def generar_datos(n=1_000_000, random_state=42):
    rng = np.random.default_rng(random_state)
    columns = ["A", "B", "C", "Y"]
    df = pd.DataFrame(
        data=np.full(shape=(n, len(columns)), fill_value=0, dtype=int),
        columns=columns,
    )

    # 990,000 registros con todas las condiciones en 1
    indices_todas_condiciones = list(df.sample(990_000, random_state=random_state).index)
    df.loc[indices_todas_condiciones, ["A", "B", "C", "Y"]] = 1

    # 7,000 con luz y esfera, pero no superficie ni sombra
    indices_luz_esfera = list(
        df.query("A==0 and B==0 and C==0 and Y==0").sample(7_000, random_state=random_state).index
    )
    df.loc[indices_luz_esfera, ["A", "B"]] = 1

    # 1,500 con luz y superficie, pero no esfera ni sombra
    indices_luz_superficie = list(
        df.query("A==0 and B==0 and C==0 and Y==0").sample(1_500, random_state=random_state).index
    )
    df.loc[indices_luz_superficie, ["A", "C"]] = 1

    return df


# ===========================================================
# 2. Resumen de datos
# ===========================================================
def resumen_datos(df: pd.DataFrame):
    resumen = df.value_counts().reset_index()
    resumen.columns = ["Luz(A)", "Esfera(B)", "Superficie(C)", "Sombra(Y)", "Cantidad"]
    return resumen


# ===========================================================
# 3. Visualización del modelo
# ===========================================================
def plot_model(model, pos: dict = None, tam_fig: tuple = (10, 8), diam_nodo: int = 5000):
    fig, ax = plt.subplots(figsize=tam_fig)
    plt.box(False)
    nx.draw_networkx(
        nx.DiGraph(model.edges),
        with_labels=True,
        pos=pos,
        node_color=["C{}".format(i) for i in range(len(pos))],
        node_size=diam_nodo,
        arrowsize=25,
        ax=ax,
    )
    st.pyplot(fig)



# ===========================================================
# 4. Modelos Causales Estructurales
# ===========================================================
def discover_model(data: pd.DataFrame):
    pos = {"A": np.array([0, 1]), "B": np.array([-1, 0]),
           "C": np.array([1, 0]), "Y": np.array([0, -1])}
    tam_fig = (10, 8)

    pc_estimator = PC(data)
    model = pc_estimator.estimate()

    graph = nx.DiGraph(model.edges())
    fig, ax = plt.subplots(figsize=tam_fig)
    nx.draw(
        graph,
        pos=pos,
        with_labels=True,
        node_size=3000,
        node_color="lightblue",
        font_size=15,
        font_weight="bold",
        edge_color="gray",
        ax=ax
    )
    ax.set_title("Causal Discovery Model", fontsize=20)

    return model, fig



def modelo_proporcionado():
    pos = {"A": np.array([0, 1]), "B": np.array([-1, 0]),
           "C": np.array([1, 0]), "Y": np.array([0, -1])}
    tam_fig = (10, 8)

    modelo = DiscreteBayesianNetwork  ([("A", "Y"), ("B", "Y"), ("C", "Y")])
    return modelo, pos, tam_fig


# ===========================================================
# 5. Distribuciones de probabilidad condicional (CPDs)
# ===========================================================
def generar_CPD(modelo: DiscreteBayesianNetwork    , data: pd.DataFrame):
    modelo.fit(data, estimator=MaximumLikelihoodEstimator)
    modelo.check_model()
    cpds = modelo.get_cpds()
    for cpd in cpds:
        print(cpd)
    return cpds

def cpd_to_dataframe(cpd):
    """
    Convierte un TabularCPD de pgmpy en un DataFrame legible y estético
    """
    variable = cpd.variable
    estados_variable = cpd.state_names[variable]

    # Si no tiene padres → solo una distribución
    if not cpd.variables[1:]:
        df = pd.DataFrame(
            cpd.values.reshape(-1, 1),
            index=estados_variable,
            columns=[f"P({variable})"]
        )
        return df

    # Si tiene padres → necesitamos expandir combinaciones
    padres = cpd.variables[1:]
    estados_padres = [cpd.state_names[p] for p in padres]
    combinaciones = list(itertools.product(*estados_padres))

    # Aplanar cpd.values y construir DataFrame
    valores = cpd.values.reshape(len(estados_variable), -1)
    multi_index = pd.MultiIndex.from_tuples(combinaciones, names=padres)

    df = pd.DataFrame(
        valores,
        index=estados_variable,
        columns=multi_index
    )
    return df

# ===========================================================
# 6. Consultas tipo "What if..."
# ===========================================================
def consultar_modelo(modelo: DiscreteBayesianNetwork  , variables, evidencia=None):
    infer = VariableElimination(modelo)
    return infer.query(variables=variables, evidence=evidencia)

def consultar_y_formatear(modelo, df, variables, evidencia=None, descripcion=""):
    """
    Usa consultar_modelo y devuelve resultados en número absoluto y porcentaje.
    """
    resultado = consultar_modelo(modelo, variables=variables, evidencia=evidencia)
    total = df.shape[0]

    # Probabilidad de que Y=1 (sombra presente)
    prob_sombra = resultado.values[1]
    cantidad = int(prob_sombra * total)
    porcentaje = round(prob_sombra * 100, 2)

    return {
        "descripcion": descripcion,
        "cantidad": cantidad,
        "porcentaje": porcentaje
    }

def consultas_whatif(modelo: DiscreteBayesianNetwork  , df: pd.DataFrame):
    consulta_1 = consultar_modelo(modelo, variables=["Y"], evidencia=None)
    print(f"Cantidad de sombras detectadas: {int(consulta_1.values[1] * df.shape[0])}\n")

    consulta_2 = consultar_modelo(modelo, variables=["Y"], evidencia={"B": 0})
    print(f"Número de sombras si no hubiera habido esfera: {int(consulta_2.values[1] * df.shape[0])}\n")

    consulta_3 = consultar_modelo(modelo, variables=["Y"], evidencia={"A": 0})
    print(f"Número de sombras si no hubiera habido luz: {int(consulta_3.values[1] * df.shape[0])}\n")


# ===========================================================
# 7. Causalidad: ATE e Intervenciones
# ===========================================================
from pgmpy.models import DiscreteBayesianNetwork
from causalinference import CausalModel
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
import pandas as pd

def do_intervention(modelo, variable, valor, objetivo):
    """
    Realiza la intervención do(variable=valor) en el modelo y devuelve
    la distribución resultante sobre el objetivo en formato DataFrame.
    """
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination

    # Crear nueva CPD forzada (do)
    if valor == 0:
        valores = [[1.0], [0.0]]   # 2 filas, 1 columna
    else:
        valores = [[0.0], [1.0]]   # 2 filas, 1 columna

    cpd = TabularCPD(variable=variable, variable_card=2, values=valores)

    # Clonar modelo y reemplazar CPD
    modelo_do = modelo.copy()
    modelo_do.remove_cpds(variable)
    modelo_do.add_cpds(cpd)

    infer = VariableElimination(modelo_do)
    resultado = infer.query(variables=[objetivo])

    # Pasar a DataFrame
    estados = resultado.state_names[objetivo]
    df = pd.DataFrame(
        resultado.values.reshape(-1, 1),
        index=estados,
        columns=[f"P({objetivo} | do({variable}={valor}))"]
    )
    return df




def causalidad(df: pd.DataFrame, modelo=None, tipo="ate"):
    """
    tipo puede ser:
    - "ate": calcula Average Treatment Effect con CausalModel
    - "do" : calcula intervenciones con VariableElimination sobre el modelo pgmpy
    """
    if tipo == "ate":
        # Variables
        Y = df.loc[:, "Y"].values
        B = df.loc[:, "B"].values
        confounders = df.drop(columns=["B", "Y"]).values

        model = CausalModel(Y, B, confounders)

        # Estimación OLS
        model.est_via_ols(adj=0)

        resultado = {
            "ATE": model.estimates.ate,
            "ATT": model.estimates.att,
            "ATC": model.estimates.atc,
            "Resumen": model.estimates.summary()
        }
        return resultado

    elif tipo == "do":
        if modelo is None:
            raise ValueError("Se requiere un modelo BayesianNetwork para calcular intervenciones (do-calculus).")

        df_do0 = do_intervention(modelo, variable="B", valor=0, objetivo="Y")
        df_do1 = do_intervention(modelo, variable="B", valor=1, objetivo="Y")

        return {"do(B=0)": df_do0, "do(B=1)": df_do1}

    else:
        raise ValueError("El parámetro 'tipo' debe ser 'ate' o 'do'.")




# ===========================================================
# 8. Detección de sombras en imágenes
# ===========================================================
import cv2
from skimage import io, color, segmentation


def detectar_sombra_threshold(image_path: str, min_area: int = 20):
    """
    Detección de sombra usando threshold binario fijo
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_image, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 20:
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)

    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image_rgb, cmap="gray" if len(image_rgb.shape) == 2 else None)
    ax.set_title("Detección de Sombra")
    ax.axis("off")
    st.pyplot(fig)


def detectar_sombra_adaptativo(image_path: str, min_area: int = 100):
    """
    Detección de sombra usando umbral adaptativo
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)

    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image_rgb, cmap="gray" if len(image_rgb.shape) == 2 else None)
    ax.set_title("Umbral Adaptativo")
    ax.axis("off")
    st.pyplot(fig)


def detectar_sombra_felzenszwalb(image_path: str, scale: int = 100, sigma: float = 0.5, min_size: int = 50, umbral: float = 0.1):
    """
    Detección de sombra usando segmentación de Felzenszwalb
    """
    from skimage import io, color, segmentation

    image = io.imread(image_path)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    gray_image = color.rgb2gray(image)

    segments = segmentation.felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
    segment_means = []
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        segment_means.append(np.mean(gray_image[mask]))

    shadow_segments = np.where(np.array(segment_means) < 0.1)
    shadow_mask = np.isin(segments, shadow_segments)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.contour(shadow_mask, colors="red", linewidths=2)
    ax.set_title("Segmentación Felzenszwalb")
    st.pyplot(fig)
