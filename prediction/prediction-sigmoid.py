# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:25:06 2024

@author: Gaspar Sulub
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el modelo SVM previamente entrenado
sigmoid_kernel = joblib.load('svm_model_sigmoid.joblib')  
support_vectors = joblib.load('svm_support_vectors_sigmoid.joblib')  

# Leer los nuevos datos para hacer predicciones (conjunto de datos similar)
df_predict = pd.read_csv('dataset/my-dataset-prediction.csv')  

# Realizar preprocesamiento similar al conjunto de entrenamiento
df_predict.replace([np.inf, -np.inf], np.nan, inplace=True)

# Borrar valores perdidos
df_predict.drop(['Timestamp'], axis=1, inplace=True)
df_predict.dropna(inplace=True)
df_predict.dropna(axis=1, inplace=True) 

# Eliminar filas duplicadas
print("-------Duplicados------")
print("Encontrados")
print(df_predict.duplicated().sum())

df_predict.drop_duplicates(inplace=True)
print("-------Quedan------")
print(df_predict.duplicated().sum())
df_predict.drop_duplicates(inplace=True)

# Etiquetar como maligno o benigno
label_encoder = LabelEncoder()
df_predict['Label'] = label_encoder.fit_transform(df_predict['Label'])

# Extraer características y etiquetas del conjunto de datos similar


# Normalizar características del conjunto de datos similar usando el mismo MinMaxScaler
feature_list = ['Dst Port','Protocol','Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max',
    'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
    'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
    'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
    'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
    'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio',
    'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
    'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts',
    'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean',
    'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']


# Crear una instancia de MinMaxScaler con la lista de características
min_max_scaler = MinMaxScaler(feature_range=(0, 1), copy=True)

# Ajustar y transformar los datos usando el MinMaxScaler solo para las características especificadas
df_predict[feature_list] = min_max_scaler.fit_transform(df_predict[feature_list])

# Realizar predicciones en el conjunto de datos similar usando el modelo previamente entrenado
predictions_weighted = sigmoid_kernel.predict(df_predict.drop("Label", axis=1).values)

# Calcular y mostrar métricas de evaluación en el conjunto de datos similar
accuracy_weighted = accuracy_score(df_predict['Label'], predictions_weighted)
precision_weighted = precision_score(df_predict['Label'], predictions_weighted, average='weighted', zero_division=1)
recall_weighted = recall_score(df_predict['Label'], predictions_weighted, average='weighted', zero_division=1)
f1_weighted = f1_score(df_predict['Label'], predictions_weighted, average='weighted', zero_division=1)

print(f'Accuracy: {accuracy_weighted}')
print(f'Precision: {precision_weighted}')
print(f'Recall: {recall_weighted}')
print(f'F1 Score: {f1_weighted}')

# Calcular y mostrar la matriz de confusión en el conjunto de datos similar
conf_matrix_weighted = confusion_matrix(df_predict['Label'], predictions_weighted)
conf_matrix_weighted_df = pd.DataFrame(conf_matrix_weighted, index=['Actual Benign', 'Actual Malicious'], columns=['Predicted Benign', 'Predicted Malicious'])
print('\nConfusion Matrix:\n', conf_matrix_weighted_df)

# Plot matriz de confusión usando seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_weighted_df, annot=True, cmap='Blues', annot_kws={"size": 12})
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.tight_layout()
plt.show()

# Guardar métricas de evaluación en un archivo Joblib
# Crear un DataFrame para almacenar las métricas de evaluación
metrics_data = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [accuracy_weighted, precision_weighted, recall_weighted, f1_weighted]
})

# Guardar las métricas de evaluación en un archivo Joblib
joblib.dump(metrics_data, 'predictions_metrics_sigmoid.joblib')

# Guardar la matriz de confusión en un archivo Joblib
joblib.dump(conf_matrix_weighted_df, 'prediction_matrix_sigmoid.joblib')

# Agregar una columna de predicciones al DataFrame
df_predict['Predictions'] = predictions_weighted

# Guardar el DataFrame con las predicciones en un archivo CSV
df_predict.to_csv('predictions_sigmoid.csv', index=False)



