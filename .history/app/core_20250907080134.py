# app/core.py
import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def construir_red():
    """
    Construye la red bayesiana fija del experimento.
    Retorna un modelo BayesianNetwork ya configurado.
    """

    # Definir estructura de la red (ejemplo adaptado del notebook)
    modelo = BayesianNetwork([("Luz", "Sombra"), ("Objeto", "Sombra")])

    # Definir CPDs (estos valores puedes afinarlos según tu experimento real)
    cpd_luz = TabularCPD("Luz", 2, [[0.7], [0.3]])
    cpd_objeto = TabularCPD("Objeto", 2, [[0.6], [0.4]])
    cpd_sombra = TabularCPD(
        "Sombra", 2,
        [[0.9, 0.6, 0.7, 0.1],   # P(Sombra=0|...)
         [0.1, 0.4, 0.3, 0.9]],  # P(Sombra=1|...)
        evidence=["Luz", "Objeto"],
        evidence_card=[2, 2]
    )

    modelo.add_cpds(cpd_luz, cpd_objeto, cpd_sombra)

    return modelo


def consultar(modelo, objetivo, evidencia):
    """
    Ejecuta inferencia sobre la red bayesiana.
    - objetivo: variable a consultar
    - evidencia: diccionario con la evidencia observada
    Retorna la distribución de probabilidad de la variable objetivo.
    """
    infer = VariableElimination(modelo)
    resultado = infer.query(variables=[objetivo], evidence=evidencia)
    return resultado
