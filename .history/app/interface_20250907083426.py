import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from app.core import generar_datos, resumen_datos, descubrir_causalidad

st.set_page_config(page_title="Causal Inference Shadow Detection", layout="wide")

st.title("🔍 Shadow Detection - Causal Inference")

# Paso 1: Generar datos
st.header("1. Generación de datos")
n = st.slider("Cantidad de muestras", 100, 5000, 1000, step=100)
df = generar_datos(n)

st.success("Datos generados correctamente ✅")

# Mostrar preview
st.subheader("Vista previa de los datos")
st.dataframe(df.head())

# Paso 2: Resumen de datos
st.header("2. Resumen de los datos")
resumen = resumen_datos(df)
st.write(resumen)

# Gráfico: histogramas
st.subheader("Distribuciones")
fig, ax = plt.subplots(figsize=(8,4))
df.hist(ax=ax)
st.pyplot(fig)

# Paso 3: Descubrimiento causal
st.header("3. Descubrimiento Causal")
grafo = descubrir_causalidad(df)

st.subheader("Grafo descubierto")
fig, ax = plt.subplots()
pos = nx.spring_layout(grafo)
nx.draw(grafo, pos, with_labels=True, node_color="lightblue", ax=ax, arrows=True)
st.pyplot(fig)
