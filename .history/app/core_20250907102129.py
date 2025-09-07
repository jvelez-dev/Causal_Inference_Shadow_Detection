# app/core.py
import numpy as np
import pandas as pd
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
    return df.value_counts().reset_index().rename({0: "Count"}, axis=1)


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
    plt.figure(figsize=tam_fig)
    nx.draw(
        graph,
        pos=pos,
        with_labels=True,
        node_size=3000,
        node_color="lightblue",
        font_size=15,
        font_weight="bold",
        edge_color="gray",
    )
    plt.title("Causal Discovery Model", fontsize=20)
    plt.show()

    return model


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

def do_intervention(modelo, variable: str, valor: int, objetivo: str):
    """
    Ejecuta una intervención do(variable=valor) en un modelo bayesiano
    y consulta la distribución del objetivo.
    """
    # Copiar modelo
    intervened_model = modelo.copy()

    # Eliminar aristas entrantes a la variable intervenida
    parents = list(intervened_model.predecessors(variable))
    for p in parents:
        intervened_model.remove_edge(p, variable)

    # Definir CPD determinista para la intervención
    if valor == 0:
        values = [[1], [0]]  # P(variable=0)=1
    else:
        values = [[0], [1]]  # P(variable=1)=1

    cpd = TabularCPD(variable=variable, variable_card=2, values=values)
    intervened_model.add_cpds(cpd)

    # Inferencia sobre el modelo intervenido
    infer = VariableElimination(intervened_model)
    resultado = infer.query(variables=[objetivo])
    return resultado


def causalidad(df: pd.DataFrame, modelo=None, tipo="ate"):
    """
    tipo puede ser:
    - "ate": calcula Average Treatment Effect con CausalModel
    - "do" : calcula intervenciones con VariableElimination sobre el modelo pgmpy
    """
    if tipo == "ate":
        # Definir outcome, tratamiento y confounders
        Y = df.loc[:, "Y"].values
        B = df.loc[:, "B"].values
        confounders = df.drop(columns=["B", "Y"]).values

        model = CausalModel(Y, B, confounders)

        # Estimación OLS
        model.est_via_ols(adj=0)
        estimaciones = model.estimates

        resultado = {
            "estimaciones": estimaciones,
            "resumen_stats": model.summary_stats
        }
        return resultado

    elif tipo == "do":
        if modelo is None:
            raise ValueError("Se requiere un modelo BayesianNetwork para calcular intervenciones (do-calculus).")
        
        resultado = do_intervention(modelo, variable="B", valor=0, objetivo="Y")
        return resultado

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
