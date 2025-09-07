import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from app.core import generar_datos, resumen_datos, descubrir_causalidad
# (Luego agregaremos también funciones SCM, inferencia, ATE, etc.)

st.set_page_config(page_title="Causal Inference Shadow Detection", layout="wide")

st.title("🔍 Shadow Detection - Causal Inference")

# Menú lateral
menu = st.sidebar.radio(
    "Selecciona una opción:",
    (
        "1. Generación del conjunto de datos observables",
        "2. Generación del SCM",
        "3. Inferencia causal",
        "4. Causalidad (ATE e Intervenciones)",
        "5. Visualizaciones"
    )
)

# 1. Generación de datos observables
if menu.startswith("1"):
    st.header("1. Generación del conjunto de datos observables")
    n = st.slider("Cantidad de muestras", 100, 5000, 1000, step=100)
    df = generar_datos(n)

    st.subheader("Resumen de datos (frecuencias)")
    st.dataframe(resumen_datos(df))

    # Visualizar grafo original (modelo dado)
    st.subheader("Modelo proporcionado")
    grafo_prov = nx.DiGraph([("Luz", "Sombra"), ("Objeto", "Sombra")])
    fig, ax = plt.subplots()
    nx.draw(grafo_prov, with_labels=True, node_color="lightblue", ax=ax, arrows=True)
    st.pyplot(fig)

# 2. SCM
elif menu.startswith("2"):
    st.header("2. Generación del SCM")
    n = st.slider("Cantidad de muestras", 100, 5000, 1000, step=100)
    df = generar_datos(n)

    st.subheader("Modelo proporcionado")
    grafo_prov = nx.DiGraph([("Luz", "Sombra"), ("Objeto", "Sombra")])
    fig, ax = plt.subplots()
    nx.draw(grafo_prov, with_labels=True, node_color="lightblue", ax=ax, arrows=True)
    st.pyplot(fig)

    st.subheader("Modelo descubierto (PC)")
    grafo_pc = descubrir_causalidad(df)
    fig, ax = plt.subplots()
    pos = nx.spring_layout(grafo_pc, seed=42)
    nx.draw(grafo_pc, pos, with_labels=True, node_color="lightgreen", ax=ax, arrows=True)
    st.pyplot(fig)

# 3. Inferencia causal
elif menu.startswith("3"):
    st.header("3. Inferencia causal")
    st.write("Aquí iría la construcción de la CPD y consultas como:")
    st.markdown("""
    - ¿Cuántas sombras hay detectadas?  
    - ¿Qué pasaría con la sombra si no hubiera habido esfera?  
    - ¿Qué pasaría con la sombra si no hubiera habido luz?  
    """)

# 4. Causalidad
elif menu.startswith("4"):
    st.header("4. Causalidad (ATE e Intervenciones)")
    st.write("Aquí calcularemos ATE y efectos de intervenciones usando do-calculus.")

# 5. Visualizaciones
elif menu.startswith("5"):
    st.header("5. Visualizaciones")
    st.write("Aquí se mostrarán gráficos adicionales de los datos y el modelo.")
