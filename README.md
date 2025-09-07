# 🌑 Causal Inference Shadow Detection

> *Un entorno interactivo para explorar causalidad entre objetos y sombras*

Causal Inference Shadow Detection es un sistema interactivo desarrollado en Python con Streamlit, diseñado para simular entornos observacionales, descubrir estructuras causales y realizar inferencia mediante regresión estadística y do-calculus.

Este entorno permite a investigadores y estudiantes explorar la relación causal entre variables como Luz (A), Esfera (B), Superficie (C) y Sombra (Y), con aplicaciones en visión por computador, robótica y física computacional.

La interfaz intuitiva guía al usuario a través de cuatro etapas clave: generación de datos, descubrimiento causal, comparación con modelo de referencia e inferencia cuantitativa.

---

## 🚀 Características

- ✅ Generación automática de datasets observacionales (1000+ registros).
- ✅ Descubrimiento causal con algoritmo PC y visualización de DAGs.
- ✅ Inferencia causal cuantitativa: ATE, ATT, ATC con `causalinference`.
- ✅ Consultas de intervención (`do(B=0)`, `do(B=1)`) con `pgmpy`.
- ✅ Comparación visual y numérica entre modelo descubierto y modelo de referencia.
- ✅ Detección de sombras en imágenes con 3 métodos (Threshold, Adaptativo, Felzenszwalb).

---

## ⚙️ Requisitos

- Python 3.12 o superior
- Librerías en `requirements.txt` (instalables con `pip install -r requirements.txt`)
- Streamlit

---

## 🛠️ Instalación y Ejecución

```bash
# Clonar repositorio
git clone [https://github.com/tu-usuario/CausalInference_ShadowDetection.git](https://github.com/jvelez-dev/Causal_Inference_Shadow_Detection.git)
cd CausalInference_ShadowDetection

# Crear y activar entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar app
streamlit run app/interface.py
