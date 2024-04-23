# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:43:57 2023

@author: Gaspar Sulub
"""

import joblib

# Cargar el modelo entrenado desde el archivo
loaded_model = joblib.load('svm_model_linear.joblib')

# Cargar las métricas desde el archivo
metrics_linear = joblib.load('svm_metrics_linear.joblib')

# Mostrar las métricas
print("Metrics from SVM Training with Linear Kernel:")
print("Accuracy:", metrics_linear['accuracy'])
print("Precision:", metrics_linear['precision'])
print("Recall:", metrics_linear['recall'])
print("F1 Score:", metrics_linear['f1'])
