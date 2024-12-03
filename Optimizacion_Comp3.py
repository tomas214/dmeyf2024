# Importación de librerías
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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


# Lista de columnas a mantener
columns_to_select = [
    'numero_de_cliente', 'foto_mes', 'clase_ternaria',
    'mrentabilidad_annual', 'mcuentas_saldo', 'cliente_edad',
    'mcuenta_corriente', 'Visa_fechaalta', 'mactivos_margen',
    'Master_Fvencimiento', 'Master_fechaalta', 'mrentabilidad',
    'ctrx_quarter', 'mcaja_ahorro', 'Visa_Fvencimiento',
    'Visa_mfinanciacion_limite', 'cliente_antiguedad',
    'Master_mfinanciacion_limite', 'Visa_mpagominimo', 'Visa_mpagospesos',
    'cproductos', 'chomebanking_transacciones', 'mtransferencias_recibidas',
    'Visa_mconsumospesos', 'Visa_cconsumos', 'Visa_msaldopesos',
    'mpasivos_margen'
]

# Subsetear las columnas del DataFrame
data_subset = data[columns_to_select]

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


# Convertir 'foto_mes' a formato cadena
data['foto_mes'] = data['foto_mes'].astype(str)

# Guardo que quede por ahí la base para predecir luego septiembre 2021
data_Sept = data[data['foto_mes'] == '202109']

# Eliminar meses innecesarios (enero sólo para
# lags deltalag, y junio se rompe, 
# y el mes a preecir que ya lo guarde en otro lado)
data = data[~data['foto_mes'].isin(['202001', '202006', '202109'])]



# Chequeo
niveles_foto_mes = data['foto_mes'].unique()
print(niveles_foto_mes)

# Asignación de pesos a las clases
data['clase_peso'] = 1.0
data.loc[data['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00002
data.loc[data['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

# Crear la clase binaria para el modelo
data['clase_binaria'] = np.where(data['clase_ternaria'] == 'BAJA+2', 1, 0)

#Guardo esa base
data.to_csv(r'C:\Users\tomas\Desktop\DM\DMEyF\datasets\data.csv', index=False)

# Preparar conjuntos de datos
mes_train = ['202009', '202010', '202011', '202012', '202101', '202102', '202103', '202104']
mes_val = "202105"
mes_test = "202106"


train_data = data[data['foto_mes'].isin(mes_train)].drop(['foto_mes', 'numero_de_cliente'], axis=1)
val_data = data[data['foto_mes'] == mes_val].drop(['foto_mes', 'numero_de_cliente'], axis=1)
test_data = data[data['foto_mes'] == mes_test].drop(['foto_mes', 'numero_de_cliente'], axis=1)

# Separar características (X), etiquetas (y) y pesos (w)
X_train = train_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria'], axis=1)
y_train = train_data['clase_binaria']
w_train = train_data['clase_peso']

X_val = val_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria'], axis=1)
y_val = val_data['clase_binaria']
w_val = val_data['clase_peso']

X_test = test_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria'], axis=1)
y_test = test_data['clase_binaria']
w_test = test_data['clase_peso']


test_data.head()


# Crear datasets para LightGBM con feature_pre_filter=False
lgb_train = lgb.Dataset(X_train, label=y_train, weight=w_train, params={"feature_pre_filter": False})
lgb_val = lgb.Dataset(X_val, label=y_val, weight=w_val, params={"feature_pre_filter": False})

# Generar semillas aleatorias fijas para Optuna
random_seeds = np.random.randint(1, 10**6, size=500)

def objective(trial):
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
        'verbose': -1,
        'seed': seed,
    }

    # Entrenar el modelo
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=100)],
        feval=lambda y_pred, data: (
            'ganancia',
            calcular_ganancia(data.get_label(), y_pred, data.get_weight()),
            True
        )
    )

    # Almacenar número de iteraciones
    trial.set_user_attr('n_boost_rounds', model.best_iteration)

    # Predecir sobre el conjunto de test (septiembre)
    y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)
    ganancia_test = calcular_ganancia(y_test, y_pred_test, w_test)

    return ganancia_test

# Configurar y ejecutar Optuna
storage_name = "sqlite:///optimization_lgbm.db"
study_name = "exp_lgbm_optimized_Piola_2"

study = optuna.create_study(
    direction="maximize",
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True
)

study.optimize(objective, n_trials=450)


# Filtrar solo objetos pickleables
pickleable_globals = {key: value for key, value in globals().items() if not key.startswith('__') and isinstance(value, (int, float, str, list, dict, tuple, set))}

# Guardar los objetos filtrados
with open(r'C:\Users\tomas\Desktop\DM\DMEyF\datasets\entorno_completo.pkl', 'wb') as f:
    pickle.dump(pickleable_globals, f)
