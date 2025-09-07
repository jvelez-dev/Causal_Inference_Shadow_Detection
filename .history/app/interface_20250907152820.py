import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from app.core import (
    generar_datos,
    resumen_datos,
    plot_model,
    discover_model,
    modelo_proporcionado,
    generar_CPD,
    cpd_to_dataframe,
    consultar_modelo,
    consultar_y_formatear,
    causalidad,
    detectar_sombra_threshold,
    detectar_sombra_adaptativo,
    detectar_sombra_felzenszwalb,
)

# ===========================================================
# Interfaz principal con menú
# ===========================================================
# --- HEADBAR FIJA ---
st.markdown("""
    <style>
    .headbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #0d6efd;
        padding: 12px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: white;
        font-size: 18px;
        font-weight: bold;
        z-index: 9999;
    }
    .headbar .title {
        display: flex;
        align-items: center;
    }
    .headbar img {
        height: 30px;
        margin-right: 10px;
    }
    .stApp {
        margin-top: 70px; /* Para no tapar el contenido */
    }
    </style>

    <div class="headbar">
        <div class="title">
            <img src="https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg">
            Causal Inference - Shadow Detection
        </div>
        <div class="menu">
            Detección de Sombra Usando Inferencia Causal
        </div>
    </div>
""", unsafe_allow_html=True)

st.sidebar.title("Menú de operaciones")

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
    st.header("Generación del conjunto de datos observables")
    st.markdown("""
                ### Descripción
                Este conjunto de datos simula observaciones con las siguientes variables:
                - **A**: Luz  
                - **B**: Esfera  
                - **C**: Superficie  
                - **Y**: Sombra

                De los **1,000,000 registros**, el **99%** cumple con todas las condiciones para generar sombra. 
                El resto corresponde a casos especiales.
                """)
    df = generar_datos()

    st.subheader("Resumen de los datos")
    st.dataframe(resumen_datos(df))

# ===========================================================
# 2. Generación del SCM
# ===========================================================
elif menu.startswith("2"):
    st.header("Modelo Estructural Causal (SCM)")
    st.markdown("""
                    Una red bayesiana es el equivalente a un SCM, dado que en este caso conocemos las variables y sus relaciones en la imagen. Sabemos que:
                    1. A,B y C son independientes
                    2. Y es dependiente de A, B y C

                    Por lo cual, la probabilidad de Y está condicionada a los eventos A,B y C, es decir P(Y|A,B,C).
                    Trataremos de llegar a este modelo, según nuestro "expert knowledge", utilizando el descubrimiento causal.
                """)
    df = generar_datos()

    st.subheader("Causal discovery")
    st.markdown("""
                    Intentaremos inferir el SCM a partir de los datos; en otras palabras, el descubrimiento causal es la tarea de descubrir el modelo causal con base en la data. Para esta iferencia usaremos el algoritmo NOTEARS (Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for Structure learning)
                """)
    modelo_cd, fig_cd = discover_model(df)
    st.pyplot(fig_cd)

    st.subheader("Modelo ajustado - Acíclico y sin Factores de Confusión")
    modelo, _, _ = modelo_proporcionado()
    pos = {"A": [0, 1], "B": [-1, 0], "C": [1, 0], "Y": [0, -1]}
    plot_model(modelo, pos=pos)


# ===========================================================
# 3. Inferencia causal
# ===========================================================
elif menu.startswith("3"):
    st.header("Inferencia causal")
    df = generar_datos()
    modelo,_,_ = modelo_proporcionado()

    st.subheader("Distribuciones de probabilidad condicional (CPDs)")
    st.markdown("""
                Las Distribuciones de Probabilidad Condicional (CPD) se usan para describir las probabilidades de la transición de los nodos en el diagrama.
                Una CPD se define para un conjunto de variables aleatorias discretas y mutuamente dependientes para mostrar las probabilidades condicionales de una sola variable con respecto a las demás, es decir, la probabilidad de cada valor posible de una variable si conocemos los valores que toman las otras variables.
                En nuestro caso lo que interesa es saber la probabilidad del outcome (Y) dados los valoes de los demás eventos (A, B y C) que intervienen en la imagen.
                - **A**: Luz  
                - **B**: Esfera  
                - **C**: Superficie  
                - **Y**: Sombra
                """)
    cpds = generar_CPD(modelo, df)

    for cpd in cpds:
        st.markdown(f"**CPD de {cpd.variable}**")
        df_cpd = cpd_to_dataframe(cpd)
        st.dataframe(df_cpd.style.format("{:.3f}"))

    st.subheader("Consultas al modelo (What if...)")

    consulta_1 = consultar_y_formatear(modelo, df, variables=['Y'], evidencia=None, descripcion="Cantidad de sombras detectadas")
    consulta_2 = consultar_y_formatear(modelo, df, variables=['Y'], evidencia={'B': 0}, descripcion="Número de sombras si no hubiera habido esfera")
    consulta_3 = consultar_y_formatear(modelo, df, variables=['Y'], evidencia={'A': 0}, descripcion="Número de sombras si no hubiera habido luz")

    # Mostrar resultados
    for consulta in [consulta_1, consulta_2, consulta_3]:
        st.write(f"**{consulta['descripcion']}**: {consulta['cantidad']} ({consulta['porcentaje']}%)")


# ===========================================================
# 4. Causalidad
# ===========================================================
elif menu.startswith("4"):
    st.header("Causalidad")
    df = generar_datos()
    modelo,_,_ = modelo_proporcionado()
    generar_CPD(modelo, df)

    st.subheader("Average Treatment Effect (ATE)")
    resultado_ate = causalidad(df, modelo, tipo="ate")
    st.write("Estimaciones:", resultado_ate["estimaciones"])
    st.write("Resumen:", resultado_ate["resumen_stats"])

    st.subheader("Intervenciones (do-calculus)")
    resultado_do = causalidad(df, modelo, tipo="do")

    # ⚡ Mostrar tablas para do(B=0) y do(B=1)
    for nombre, df_do in resultado_do.items():
        st.markdown(f"**{nombre}**")
        st.dataframe(df_do.style.format("{:.3f}"))

# ===========================================================
# 5. Visualizaciones
# ===========================================================
elif menu.startswith("5"):
    st.header("Visualizaciones")
    st.info("Aquí podrías incluir histogramas, distribuciones o gráficos adicionales.")

# ===========================================================
# 6. Detección de sombras
# ===========================================================
elif menu.startswith("6"):
    st.header("Detección de sombras en imágenes")
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
