# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:48:55 2023

@author: Gaspar Sulub
"""
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import recall_score, f1_score, accuracy_score,precision_score,  confusion_matrix, classification_report
import time


#Importar dataset
df = pd.read_csv('dataset/my-dataset.csv')
#np.set_printoptions(precision=4)
#Mostrar dataset
#df

#Remplazar valores infinitos del df
df.replace([np.inf, -np.inf], np.nan, inplace=True)

#Borrar valores perdidos
df.drop(['Timestamp'], axis=1,inplace=True)
df.dropna(inplace=True)
df.dropna(axis=1, inplace=True) 
#Desplegar labels
print(df["Label"].value_counts())
print("---------------")


#Remplazar el nombre de las etiquetas por el ataque correspondiente
#df.replace(to_replace=["FTP-BruteForce", "SSH-Bruteforce"], value="Malicious", inplace=True)

#Eliminar filas duplicadas
print("-------Duplicados------")
print("Encontrados")
print(df.duplicated().sum())

df.drop_duplicates(inplace = True)
print("-------Quedan------")
print(df.duplicated().sum())

#Balanceo del dataset
df1 = df[df["Label"] == "Benign"][:2150]
df2 = df[df["Label"] == "Malicious"][:2150]
df_equal = pd.concat([ df1,df2], axis =0)

#Etiquetar como maligno o beningno
df_equal.replace(to_replace="Benign", value=0, inplace=True)
df_equal.replace(to_replace="Malicious", value=1, inplace=True)

#Entrenamiento
train, test = train_test_split(df_equal, test_size=0.2, random_state=0)
#Desplegar columnas a entrenar
train.columns
#Info. de las columnas
train.info()

#Aplicar min-max a los datos de prueba y entrenamiento
min_max_scaler = MinMaxScaler().fit(train[['Dst Port','Protocol','Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
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
    'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']])

numerical_columns = ['Dst Port','Protocol','Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
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



train[numerical_columns] = min_max_scaler.transform(train[numerical_columns])

#train.drop(['Timestamp'], axis=1,inplace=True)
#test.drop(['Timestamp'],axis=1,inplace=True)

test[numerical_columns] = min_max_scaler.transform(test[numerical_columns])

#Desplegar cantidad de elementos
print("Data:\n")
print("Benign: " + str(df_equal["Label"].value_counts()[[0]].sum()))
print("Malicious: " + str(df_equal["Label"].value_counts()[[1]].sum()))
print("---------------")
#Cantidad de elementos a entrenar
print("Training set:\n")
print("Benign: " + str(train["Label"].value_counts()[[0]].sum()))
print("Malicious: " + str(train["Label"].value_counts()[[1]].sum()))
print("---------------")
#Cantidad de elementos de prueba
print("Test set:\n")
print("Benign: " + str(test["Label"].value_counts()[[0]].sum()))
print("Malicious: " + str(test["Label"].value_counts()[[1]].sum()))
print("---------------")


#La columna "Label" se elimina del conjunto de datos de entrenamiento (train) y se almacena en la variable y_train
y_train = np.array(train.pop("Label"))# pop removes "Label" from the dataframe
X_train = train.values
y_test = np.array(test.pop("Label")) # pop remueve "Label" del dataset
X_test = test.values
#inicio del entrenamiento 
start_time = time.time()

#Kernel e hiperparamentros
linear_kernel = svm.SVC(kernel='linear', C=1.0)

linear_kernel.fit(X_train, y_train)
#Fin del tiempo de entrenamiento
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo transcurrido durante el entrenamiento: {elapsed_time} segundos")

#Realizar predicciones en el conjunto de prueba (X_test)
predictions = linear_kernel.predict(X_test)

accuracy_linear = accuracy_score(y_test, predictions)
precision_linear = precision_score(y_test, predictions, average='weighted')
recall_linear = recall_score(y_test, predictions, average='weighted')
f1_linear = f1_score(y_test, predictions, average='weighted')


#Guarda el modelo entrenado y las métricas en archivos separados usando joblib
joblib.dump(linear_kernel, 'svm_model_linear.joblib')

metrics_linear = {
    'accuracy': accuracy_linear,
    'precision': precision_linear,
    'recall': recall_linear,
    'f1': f1_linear
}
joblib.dump(metrics_linear, 'svm_metrics_linear.joblib')

#Guardar los vectores de soporte
support_vectors = linear_kernel.support_vectors_

#Guardar los vectores de soporte en un archivo separado usando joblib
joblib.dump(support_vectors, 'svm_support_vectors_linear.joblib')


#Metricas de evaluacion 

#Calcular y mostrar la precision del modelo en el conjunto de prueba
accuracy = linear_kernel.score(X_test, y_test)
print("Presición",accuracy)
print("Presición: {:.2f} %".format(accuracy * 100))

#Matriz de cofusion
conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión')
plt.xlabel('Predicciones')
plt.ylabel('Valores Verdaderos')
#Guardar Plot
plt.savefig('matriz_linear.png')
plt.show()

#Archivo matriz de confusion
joblib.dump(conf_matrix, 'confusion_matrix_linear.joblib')

#Exactitud - accuracy
print("Exactitud")
exactitud = accuracy_score(y_test, predictions)
print(exactitud)
#F1
print("F1")
puntaje=f1_score(y_test, predictions)
print(puntaje)

#Sensibilidad
print("Sensibilidad")
sensibilidad = recall_score(y_test, predictions)
print(sensibilidad)

#Reporte
report = classification_report(y_test, predictions)
print(report)

print(conf_matrix)











