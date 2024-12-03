# Importación de librerías
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


# Modelos y validación de sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Modelado con LightGBM
import lightgbm as lgb

# Optuna para optimización de hiperparámetros
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_contour

# Para medición de tiempos
from time import time

# Guardado de modelos
import pickle


# Parámetros de ganancia
ganancia_acierto = 273000
costo_estimulo = 7000

# Función para calcular la ganancia acumulada
def calcular_ganancia(y_true, y_pred, weights):
    sorted_indices = np.argsort(y_pred)[::-1]  # Ordenar por las predicciones
    y_true = np.array(y_true)[sorted_indices]
    weights = np.array(weights)[sorted_indices]

    ganancia = np.where(y_true == 1, ganancia_acierto, -costo_estimulo)
    ganancia_acumulada = np.cumsum(ganancia * weights)
    return np.max(ganancia_acumulada)

# Función para calcular la ganancia con probabilidades
def ganancia_prob(y_pred, y_true, prop=1):
    ganancia = np.where(y_true == 1, ganancia_acierto, 0) - np.where(y_true == 0, costo_estimulo, 0)
    return ganancia[y_pred >= 0.025].sum() / prop


# Arranca script

#Cargo Dataset
dataset_path = r"C:\Users\tomas\Desktop\DM\DMEyF\datasets"  # Usar r para indicar una cadena cruda
dataset_file = "DatosComp3.csv"
data = pd.read_csv(dataset_path + "\\" + dataset_file)

data.head()

# Ordenar el dataframe por cliente y mes
data= data.sort_values(by=['numero_de_cliente', 'foto_mes'])

# Ver los niveles de la variable 'foto_mes' 
print(data['foto_mes'].unique())

# Calcular la proporción de cada clase en la columna 'clase_ternaria'
print(data.groupby('clase_ternaria')['numero_de_cliente'].nunique())


# Excluir columnas específicas
exclude_cols = ['numero_de_cliente', 'foto_mes', 'clase_ternaria']

# Identificar las columnas para calcular Lag, dLag y percentiles
cols_to_process = [col for col in data.columns if col not in exclude_cols]

# Calcular Lag, dLag y percentiles
for col in cols_to_process:
    # Calcular Lag (valor del mes anterior)
    data[f'lag_{col}'] = data.groupby('numero_de_cliente')[col].shift(1)

    # Calcular dLag (diferencia entre valor actual y el del mes anterior)
    data[f'dlag_{col}'] = data[col] - data[f'lag_{col}']

    # Calcular Percentiles (sin incluir Lag ni dLag)
    data[f'percentile_{col}'] = data[col].rank(pct=True)

data.head()



# Eliminar meses innecesarios (enero sólo para
# lags deltalag, y junio se rompe, 
# y saco tambien julio y agosto que no tienen informacion)
data['foto_mes'] = data['foto_mes'].astype(int)
data = data[~data['foto_mes'].isin([202001, 202006, 202107, 202108])]


# Chequeo
niveles_foto_mes = data['foto_mes'].unique()
print(niveles_foto_mes)

# Asignación de pesos a las clases
data['clase_peso'] = 1.0
data.loc[data['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00002
data.loc[data['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

# Crear la clase binaria para el modelo
data['clase_binaria'] = np.where(data['clase_ternaria'] == 'BAJA+2', 1, 0)


with open(r'C:\Users\tomas\Desktop\DM\DMEyF\datasets\entorno_completo.pkl', 'rb') as f:
    loaded_objects = pickle.load(f) #Optimización en otro script

print(loaded_objects.keys())


best_params = loaded_objects.get('best_params')
num_boost_rounds = loaded_objects.get('num_boost_rounds')


# Definir los meses de entrenamiento y test
mes_train = [202002, 202003, 202004, 202005, 202007, 202008, 
             202009, 202010, 202011, 202012, 202101, 202102, 
             202103, 202104, 202105, 202106]
mes_test = 202109  # Septiembre 2021

# Datos de entrenamiento (excluyendo 'foto_mes' y 'numero_de_cliente')
train_data = data[data['foto_mes'].isin(mes_train)].drop(['foto_mes', 'numero_de_cliente'], axis=1)

# Variables para entrenamiento y pesos
X_train = train_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria'], axis=1)
y_train = train_data['clase_binaria']
w_train = train_data['clase_peso']

# Crear el dataset de LightGBM
lgb_train = lgb.Dataset(X_train, label=y_train, weight=w_train)

# Entrenamiento del modelo final sin validación
model_final = lgb.train(
    best_params,  # Los mejores parámetros obtenidos previamente
    lgb_train,
    num_boost_round=num_boost_rounds
)



# Filtrar los datos para la predicción de septiembre 2021
septiembre_data = data[data['foto_mes'] == mes_test].drop(['foto_mes', 'numero_de_cliente', 'clase_ternaria', 'clase_peso', 'clase_binaria'], axis=1)

# Asegurarse de que solo haya columnas numéricas para hacer las predicciones
septiembre_data = septiembre_data.select_dtypes(include=['float64', 'int64'])

# Realizar las predicciones con el modelo final
septiembre_pred = model_final.predict(septiembre_data, num_iteration=model_final.best_iteration)

# Agregar las predicciones al DataFrame
data.loc[data['foto_mes'] == mes_test, 'pred_prob'] = septiembre_pred

# Ordenar los clientes por probabilidad en orden decreciente
data_sorted = data[data['foto_mes'] == mes_test].sort_values(by='pred_prob', ascending=False)

# Lista de valores de X para los diferentes tamaños de predicción
X_values = [8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000]

# Definir la carpeta donde se guardarán los archivos
carpeta_destino = r'C:\Users\tomas\Desktop\DM\DMEyF\datasets'

# Bucle para generar los archivos con diferentes valores de X
for X in X_values:
    # Asignar clase 'BAJA+2' a los primeros X clientes
    data_sorted['Predicted'] = 0
    data_sorted.iloc[:X, data_sorted.columns.get_loc('Predicted')] = 1

    # Crear el archivo de salida con el numero_de_cliente y la predicción
    output = data_sorted[['numero_de_cliente', 'Predicted']]
    print(f"Clientes predichos como 'BAJA+2' para X={X}: {output['Predicted'].sum()}")

    # Guardar las predicciones en un CSV con un nombre dinámico basado en el valor de X
    nombre_archivo = os.path.join(carpeta_destino, f'predicciones_kaggle_{X}k.csv')  # Ruta completa al archivo
    output.to_csv(nombre_archivo, index=False)
    
    
    
    
# #Hago otra vuelta probando los mejores hiperparámetros de denicolay
# best_params2 = {
#     'num_leaves': 956,
#     'learning_rate': 0.03,
#     'min_data_in_leaf': 64,
#     'feature_fraction': 0.5,
#     'bagging_fraction': 0.7014099863277935,  # Sin cambios, ya que no se especificó un cambio
#     'bagging_freq': 7,  # Sin cambios, ya que no se especificó un cambio
#     'lambda_l1': 0.7993693192767846,  # Sin cambios, ya que no se especificó un cambio
#     'lambda_l2': 4.1031280752250225,  # Sin cambios, ya que no se especificó un cambio
#     'max_depth': 5,  # Sin cambios, ya que no se especificó un cambio
#     'min_gain_to_split': 0.45133526940862223,  # Sin cambios, ya que no se especificó un cambio
#     'max_bin': 31  # Cambiado según tu indicación
# }

# # Entrenamiento del modelo final sin validación
# model_final_Den = lgb.train(
#     best_params2,  # Los mejores parámetros obtenidos previamente
#     lgb_train,
#     num_boost_round=num_boost_rounds
# )



# # Realizar las predicciones con el modelo final
# septiembre_pred_Den = model_final_Den.predict(septiembre_data, num_iteration=model_final_Den.best_iteration)

# # Agregar las predicciones al DataFrame
# data.loc[data['foto_mes'] == mes_test, 'pred_prob_Den'] = septiembre_pred_Den

# # Ordenar los clientes por probabilidad en orden decreciente
# data_sorted_Den = data[data['foto_mes'] == mes_test].sort_values(by='pred_prob_Den', ascending=False)


# # Definir la carpeta donde se guardarán los archivos
# carpeta_destino = r'C:\Users\tomas\Desktop\DM\DMEyF\datasets'

# # Bucle para generar los archivos con diferentes valores de X
# for X in X_values:
#     # Asignar clase 'BAJA+2' a los primeros X clientes
#     data_sorted_Den['Predicted'] = 0
#     data_sorted_Den.iloc[:X, data_sorted_Den.columns.get_loc('Predicted')] = 1

#     # Crear el archivo de salida con el numero_de_cliente y la predicción
#     output = data_sorted_Den[['numero_de_cliente', 'Predicted']]
#     print(f"Clientes predichos como 'BAJA+2' para X={X}: {output['Predicted'].sum()}")

#     # Guardar las predicciones en un CSV con un nombre dinámico basado en el valor de X
#     nombre_archivo = os.path.join(carpeta_destino, f'predicciones_Den_kaggle_{X}k.csv')  # Ruta completa al archivo
#     output.to_csv(nombre_archivo, index=False)    
