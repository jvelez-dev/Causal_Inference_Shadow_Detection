# app/core.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph, draw_networkx
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import PC


# -----------------------------------------------------------
# 1. Generación de datos observables
# -----------------------------------------------------------
def generar_datos(n=1_000_000, random_state=42):
    """
    Genera un dataset con las condiciones especificadas:
    A = Luz, B = Esfera, C = Superficie, Y = Sombra
    """
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


# -----------------------------------------------------------
# 2. Resumen de datos
# -----------------------------------------------------------
def resumen_datos(df: pd.DataFrame):
    return df.value_counts().reset_index().rename({0: "Count"}, axis=1)


# -----------------------------------------------------------
# 3. Visualización del modelo
# -----------------------------------------------------------
def plot_model(model, pos: dict = None, tam_fig: tuple = (10, 8), diam_nodo: int = 5000):
    """
    Dibuja un modelo Bayesiano o un grafo de dependencias
    """
    if pos is None:
        pos = nx.spring_layout(model)

    plt.figure(figsize=tam_fig)
    plt.box(False)
    draw_networkx(
        DiGraph(model.edges),
        with_labels=True,
        pos=pos,
        node_color=["C{}".format(i) for i in range(len(pos))],
        node_size=diam_nodo,
        arrowsize=25,
    )
    plt.show()


# -----------------------------------------------------------
# 4. Modelos Causales Estructurales
# -----------------------------------------------------------
def discover_model(data: pd.DataFrame):
    """
    Estima un modelo causal a partir de los datos usando PC algorithm
    """
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
    """
    Define el modelo bayesiano estructural proporcionado por el experto
    """
    pos = {"A": np.array([0, 1]), "B": np.array([-1, 0]),
           "C": np.array([1, 0]), "Y": np.array([0, -1])}
    tam_fig = (10, 8)

    modelo = BayesianNetwork([("A", "Y"), ("B", "Y"), ("C", "Y")])
    plot_model(model=modelo, pos=pos, tam_fig=tam_fig)
    independencias = modelo.get_independencies()
    return modelo, independencias
