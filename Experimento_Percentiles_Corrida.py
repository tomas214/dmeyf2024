# Importación de librerías
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os

import json 

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

#Cargo Dataset
dataset_path = r"C:\Users\tomas\Desktop\DM\DMEyF\datasets"  # Usar r para indicar una cadena cruda
dataset_file = "data_Experimento.csv"
data = pd.read_csv(dataset_path + "\\" + dataset_file)

print(data['foto_mes'].unique())


# Asignación de pesos a las clases
data['clase_peso'] = 1.0
data.loc[data['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00002
data.loc[data['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

# Crear la clase binaria para el modelo
data['clase_binaria'] = np.where(data['clase_ternaria'] == 'BAJA+2', 1, 0)



# Definir los meses de entrenamiento y el mes de testeo
meses_train = ["202101", "202102", "202103", "202104", "202105"]

# Preparar los datasets por mes
datasets_train = {mes: data[data['foto_mes'] == int(mes)].drop(['foto_mes', 'numero_de_cliente'], axis=1) for mes in meses_train}

# Crear X e y por mes
X_y_train = {
    mes: (
        df.drop(['clase_ternaria', 'clase_peso', 'clase_binaria'], axis=1),
        df['clase_binaria'],
        df['clase_peso']
    )
    for mes, df in datasets_train.items()
}

# Generar semillas aleatorias fijas para reproducibilidad
random_seeds = np.random.randint(1, 10**6, size=500)

# Definición de la función objetivo
def objective(trial, X, y, w):
    # Definir la semilla del modelo
    seed = random_seeds[trial.number % len(random_seeds)] + trial.number

    params = {
        'objective': 'binary',
        'metric': 'None',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 50, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.25),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 5000),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        'max_depth': trial.suggest_int('max_depth', 3, 50),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
        'seed': seed
    }

    # Crear el conjunto de validación cruzada
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    ganancias = []

    # Iterar por cada partición de la validación cruzada
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_train, w_val = w.iloc[train_idx], w.iloc[val_idx]

        lgb_train = lgb.Dataset(X_train, label=y_train, weight=w_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, weight=w_val)

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_val],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=100)],
            feval=lambda y_pred, data: (
                'ganancia',
                calcular_ganancia(data.get_label(), y_pred, data.get_weight()),
                True
            )
        )

        y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
        ganancias.append(calcular_ganancia(y_val, y_pred_val, w_val))

    # Devolver la ganancia promedio
    return np.mean(ganancias)



# Ejecutar Optuna para cada mes, optimizando solo una vez por conjunto de entrenamiento
for mes in meses_train:
    X, y, w = X_y_train[mes]

    # Crear un estudio Optuna
    study = optuna.create_study(
        direction="maximize",
        study_name=f"optuna_lgbm_{mes}",
        storage=f"sqlite:///optuna_{mes}.db",
        load_if_exists=True
    )

    # Optimizar solo una vez para cada mes
    study.optimize(lambda trial: objective(trial, X, y, w), n_trials=100)

    # Guardar los mejores hiperparámetros en un archivo
    with open(f'best_params_{mes}.json', 'w') as f:
        json.dump(study.best_params, f)

    print(f"Mejores parámetros para el mes {mes}: {study.best_params}")


#Printear mejorparam de cada mes
# Imprimir los mejores hiperparámetros para cada mes
for mes in meses_train:
    with open(f'best_params_{mes}.json', 'r') as f:
        best_params = json.load(f)
    print(f"Mejores parámetros para el mes {mes}:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print("\n" + "="*50 + "\n")
    
    
    
#Hago predicciones
data['foto_mes'] = data['foto_mes'].astype(int)  
data_Test_jun = data[data['foto_mes'] == 202106]
data_Test_jun.columns

# Crear el dataset de LightGBM para junio (solo predicción)
X_test_jun = data_Test_jun.drop(['foto_mes', 'numero_de_cliente', 'clase_ternaria', 'clase_peso', 'clase_binaria'], axis=1)

# Crear un diccionario para almacenar las predicciones por separado
predicciones_separadas = {}

# Función para entrenar y predecir para cada mes
def entrenar_y_predecir(mes_train, best_params, X_y_train, random_seeds, X_test_jun):
    # Extraer los datos de entrenamiento para el mes correspondiente
    X_train_mes, y_train_mes, w_train_mes = X_y_train[mes_train]
    
    # Diccionario para almacenar las predicciones de cada modelo
    predicciones_mes = {}
    
    # Entrenar y predecir con 20 modelos con diferentes semillas
    for i in range(20):
        seed = random_seeds[i]
        best_params['seed'] = seed
        
        # Crear el dataset de LightGBM para entrenamiento
        lgb_train = lgb.Dataset(X_train_mes, label=y_train_mes, weight=w_train_mes)
        
        # Entrenar el modelo sin validación
        model = lgb.train(
            best_params,
            lgb_train,
            num_boost_round=1000
        )
        
        # Realizar predicciones en el conjunto de datos de junio
        pred_jun = model.predict(X_test_jun, num_iteration=model.best_iteration)
        
        # Guardar las predicciones en el diccionario por separado con el nombre del modelo
        modelo_name = f"{mes_train}_seed_{seed}"
        predicciones_mes[modelo_name] = pred_jun
        
        print(f"Modelo {modelo_name} entrenado y predicciones realizadas.")
    
    return predicciones_mes

# Lista de meses a procesar
meses = ['202101', '202102', '202103', '202104', '202105']



# Recorrer los meses y cargar los parámetros, entrenar y predecir
for mes_train in meses:
    # Cargar los mejores parámetros para el mes correspondiente
    with open(f'best_params_{mes_train}.json', 'r') as f:
        best_params_mes = json.load(f)
    
    # Obtener los datos de entrenamiento para el mes
    X_train_mes, y_train_mes, w_train_mes = X_y_train[mes_train]
    
    # Entrenar y obtener las predicciones para ese mes
    predicciones_mes = entrenar_y_predecir(mes_train, best_params_mes, X_y_train, random_seeds, X_test_jun)
    
    # Guardar las predicciones en el diccionario global
    predicciones_separadas[mes_train] = predicciones_mes
    print(f"Predicciones para {mes_train} completadas.")

# Una vez realizadas todas las predicciones, agregarlas al DataFrame de junio
for mes_train, predicciones_mes in predicciones_separadas.items():
    for modelo_name, pred_jun in predicciones_mes.items():
        data_Test_jun[f'pred_{modelo_name}'] = pred_jun

# Mostrar las primeras filas de data_Test_jun con todas las predicciones
data_Test_jun.head()

print(data_Test_jun.columns[-100:])


# Lista de meses a procesar
meses_entrenamiento = ['202101', '202102', '202103', '202104', '202105']

# Función para convertir las probabilidades en clases binarias (usando el umbral)
def clasificar_con_umbral(y_pred, umbral=0.025):
    # Aplicar el umbral
    return np.where(y_pred >= umbral, 1, 0)

# Usamos la misma lógica para evaluar las predicciones
ganancias = []

# Función para convertir las probabilidades en clases binarias (usando el umbral)
def clasificar_con_umbral(y_pred, umbral=0.025):
    return np.where(y_pred >= umbral, 1, 0)

# Lista de ganancias por modelo y mes
ganancias = []

# Asegúrate de que tienes la columna `clase_binaria` en tu DataFrame `data`
y_test_binaria = data_Test_jun['clase_binaria'].values  # Tomamos las etiquetas binarias de data_Test_jun

# Recorrer los modelos y meses de entrenamiento
for mes_train in meses_entrenamiento:
    # Filtrar las columnas correspondientes al mes
    pred_cols = [col for col in data_Test_jun.columns if f'pred_{mes_train}_seed' in col]

    for pred_col in pred_cols:
        # Extraer la semilla desde el nombre de la columna
        semilla = int(pred_col.split('_seed_')[1])

        # Obtener las predicciones de la columna
        y_pred_lgm = data_Test_jun[pred_col].values  # Predicciones de data_Test_jun

        # Convertir las predicciones en 1 o 0 con el umbral
        y_pred_binario = clasificar_con_umbral(y_pred_lgm, umbral=0.025)

        # Verificar que las longitudes coinciden
        print("Tamaño de y_pred_binario:", y_pred_binario.shape)
        print("Tamaño de y_test_binaria:", y_test_binaria.shape)

        # Calcular la ganancia
        ganancia = ganancia_prob(y_pred_binario, y_test_binaria)  # Aquí usamos `clase_binaria`

        # Almacenar el resultado
        ganancias.append({
            'modelo': pred_col,
            'semilla': semilla,
            'mes_entrenamiento': mes_train,
            'ganancia': ganancia
        })

# Mostrar las ganancias
for g in ganancias:
    print(g)
    
    
# Convertimos la lista de ganancias en un DataFrame
df_ganancias = pd.DataFrame(ganancias)

# Configuración para el gráfico
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# Crear el gráfico de dispersión (scatter plot) con jitter
sns.stripplot(x='mes_entrenamiento', y='ganancia', data=df_ganancias, jitter=True, dodge=True, palette="Set2", size=8)

# Personalizar el gráfico
plt.title('Ganancia por Modelo y Mes de Entrenamiento', fontsize=16)
plt.xlabel('Mes de Entrenamiento', fontsize=14)
plt.ylabel('Ganancia', fontsize=14)

# Mejorar la legibilidad
plt.xticks(rotation=45)  # Rotar las etiquetas del eje X

# Mostrar el gráfico
plt.tight_layout()
plt.show()


# Ruta para guardar el gráfico y el CSV en la carpeta correcta
local_directory = r"C:\Users\tomas\Desktop\DM\DMEyF\datasets"
plot_filename_local = os.path.join(local_directory, 'ganancia_por_mes_Perc.png')
csv_filename_local = os.path.join(local_directory, 'ganancias_Perc.csv')

# Guardar el gráfico como una imagen PNG
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.stripplot(x='mes_entrenamiento', y='ganancia', data=df_ganancias, jitter=True, dodge=True, palette="Set2", size=8)
plt.title('Ganancia por Modelo y Mes de Entrenamiento', fontsize=16)
plt.xlabel('Mes de Entrenamiento', fontsize=14)
plt.ylabel('Ganancia', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()

# Guardar el gráfico
plt.savefig(plot_filename_local)

# Guardar el DataFrame `df_ganancias` como un archivo CSV
df_ganancias.to_csv(csv_filename_local, index=False)

