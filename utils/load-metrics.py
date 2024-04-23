# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:52:33 2024

@author: Gaspar Sulub
"""

import joblib

# Cargar las métricas de evaluación desde el archivo Joblib
loaded_metrics_data = joblib.load('svm_metrics_rbf.joblib')

# Mostrar las métricas de evaluación cargadas
print("Métricas de evaluación cargadas:")
print(loaded_metrics_data)

# Cargar la matriz de confusión desde el archivo Joblib
loaded_confusion_matrix = joblib.load('confusion_matrix.joblib')

# Mostrar la matriz de confusión cargada
print("\nMatriz de confusión cargada:")
print(loaded_confusion_matrix)
