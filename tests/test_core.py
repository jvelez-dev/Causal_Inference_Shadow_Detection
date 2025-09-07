# tests/test_core.py
from app.core import construir_red, consultar

def test_modelo_basico():
    modelo = construir_red()
    assert set(modelo.nodes()) == {"Luz", "Objeto", "Sombra"}

def test_inferencia():
    modelo = construir_red()
    resultado = consultar(modelo, "Sombra", {"Luz": 1})
    assert "phi(Sombra)" in str(resultado)
