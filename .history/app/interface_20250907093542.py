import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from app.core import (
    generar_datos,
    resumen_datos,
    plot_model,
    discover_model,
    modelo_proporcionado,
    generar_CPD,
    consultar_modelo,
    causalidad,
    detectar_sombra_threshold,
    detectar_sombra_adaptativo,
    detectar_sombra_felzenszwalb,
)

# ===========================================================
# Interfaz principal con menú
# ===========================================================

st.title("Plataforma de Inferencia Causal y Detección de Sombra")
st.sidebar.title("Menú de navegación")

menu = st.sidebar.radio(
    "Selecciona una opción",
    [
        "1. Generación del conjunto de datos observables",
        "2. Generación del SCM",
        "3. Inferencia causal",
        "4. Causalidad",
        "5. Visualizaciones",
        "6. Detección de sombras en imágenes",
    ],
)

# ===========================================================
# 1. Generación del conjunto de datos observables
# ===========================================================
if menu.startswith("1"):
    st.header("1. Generación del conjunto de datos observables")
    df = generar_datos()

    st.subheader("📊 Resumen de los datos")
    st.dataframe(resumen_datos(df))

    st.subheader("📈 Modelo proporcionado")
    modelo,_,_ = modelo_proporcionado()
    pos = {"A": [0, 1], "B": [-1, 0], "C": [1, 0], "Y": [0, -1]}
    fig, ax = plt.subplots(figsize=(6, 4))
    nx.draw(
        nx.DiGraph(modelo.edges),
        pos=pos,
        with_labels=True,
        node_size=2000,
        node_color="lightblue",
        font_size=12,
        font_weight="bold",
        edge_color="gray",
        ax=ax,
    )
    st.pyplot(fig)

# ===========================================================
# 2. Generación del SCM
# ===========================================================
elif menu.startswith("2"):
    st.header("2. Generación del SCM")
    df = generar_datos()

    st.subheader("🔍 Causal discovery")
    discover_model(df)

    st.subheader("📈 Modelo proporcionado")
    modelo = proporcionar_modelo()
    pos = {"A": [0, 1], "B": [-1, 0], "C": [1, 0], "Y": [0, -1]}
    plot_model(modelo, pos=pos)

# ===========================================================
# 3. Inferencia causal
# ===========================================================
elif menu.startswith("3"):
    st.header("3. Inferencia causal")
    df = generar_datos()
    modelo = proporcionar_modelo()

    st.subheader("🔢 Distribuciones de probabilidad condicional (CPDs)")
    generar_CPD(df)

    st.subheader("❓ Consultas al modelo")
    consulta_1 = consultar_modelo(modelo, variables=["Y"], evidencia=None)
    st.write(f"Cantidad de sombras detectadas: {int(consulta_1.values[1] * df.shape[0])}")

    consulta_2 = consultar_modelo(modelo, variables=["Y"], evidencia={"B": 0})
    st.write(f"Número de sombras si no hubiera habido esfera: {int((consulta_2.values[1] * df.shape[0]))}")

    consulta_3 = consultar_modelo(modelo, variables=["Y"], evidencia={"A": 0})
    st.write(f"Número de sombras si no hubiera habido luz: {int((consulta_3.values[1] * df.shape[0]))}")

# ===========================================================
# 4. Causalidad
# ===========================================================
elif menu.startswith("4"):
    st.header("4. Causalidad")
    df = generar_datos()
    modelo = proporcionar_modelo()

    st.subheader("📊 Average Treatment Effect (ATE)")
    resultado_ate = causalidad(df, modelo, tipo="ate")
    st.write(resultado_ate)

    st.subheader("⚡ Intervenciones (do-calculus)")
    resultado_do = causalidad(df, modelo, tipo="do")
    st.write(resultado_do)

# ===========================================================
# 5. Visualizaciones
# ===========================================================
elif menu.startswith("5"):
    st.header("5. Visualizaciones")
    st.info("Aquí podrías incluir histogramas, distribuciones o gráficos adicionales.")

# ===========================================================
# 6. Detección de sombras
# ===========================================================
elif menu.startswith("6"):
    st.header("6. Detección de sombras en imágenes")
    st.write("Sube una imagen para probar los diferentes métodos de detección.")

    uploaded_file = st.file_uploader("📂 Selecciona una imagen", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        path = f"./temp_{uploaded_file.name}"
        with open(path, "wb") as f:
            f.write(uploaded_file.read())

        metodo = st.selectbox(
            "Elige un método de detección",
            ["Threshold fijo", "Umbral adaptativo", "Segmentación Felzenszwalb"],
        )

        if metodo == "Threshold fijo":
            detectar_sombra_threshold(path)
        elif metodo == "Umbral adaptativo":
            detectar_sombra_adaptativo(path)
        elif metodo == "Segmentación Felzenszwalb":
            detectar_sombra_felzenszwalb(path)
