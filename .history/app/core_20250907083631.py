import pandas as pd
import numpy as np
import networkx as nx
from pgmpy.estimators import PC

# 1. Generación de datos
def generar_datos(n=1000):
    np.random.seed(42)
    Luz = np.random.binomial(1, 0.5, n)
    Objeto = np.random.binomial(1, 0.5, n)
    Sombra = (Luz & Objeto).astype(int)

    df = pd.DataFrame({
        "Luz": Luz,
        "Objeto": Objeto,
        "Sombra": Sombra
    })
    return df

# 2. Resumen de los datos
def resumen_datos(df):
    return df.describe()

# 3. Descubrimiento causal con PC
def descubrir_causalidad(df):
    est = PC(df)
    modelo = est.estimate(return_type="dag")
    grafo = nx.DiGraph(modelo.edges())
    return grafo
