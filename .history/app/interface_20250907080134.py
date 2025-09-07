# app/interface.py
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from app.core import construir_red, consultar

st.set_page_config(page_title="Inferencia Causal - Sombras", layout="centered")

st.title("🌑 Inferencia Causal - Aparición de Sombras")
st.write("Aplicación interactiva basada en una red bayesiana predefinida.")

# Construir la red
modelo = construir_red()
variables = list(modelo.nodes())

# Seleccionar variable objetivo
objetivo = st.selectbox("Selecciona la variable objetivo:", variables)

# Seleccionar evidencia
st.subheader("Evidencia")
evidencia = {}
for var in variables:
    if var != objetivo:
        valor = st.selectbox(f"{var}:", ["Sin valor", 0, 1], key=var)
        if valor != "Sin valor":
            evidencia[var] = int(valor)

if st.button("Ejecutar inferencia"):
    resultado = consultar(modelo, objetivo, evidencia)
    st.success(f"Distribución de probabilidad para {objetivo}:")
    st.write(resultado)

    # Grafo
    st.subheader("Estructura de la Red Bayesiana")
    fig, ax = plt.subplots()
    nx.draw(modelo, with_labels=True, node_color="lightblue", ax=ax)
    st.pyplot(fig)
