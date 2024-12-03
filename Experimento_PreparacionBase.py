# Importación de librerías
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt








# Arranca script

#Cargo Dataset
dataset_path = r"C:\Users\tomas\Desktop\DM\DMEyF\datasets"  # Usar r para indicar una cadena cruda
dataset_file = "DatosParaSpyder_Comp3.csv"
data = pd.read_csv(dataset_path + "\\" + dataset_file)

data.head()

#Saco col así corre más rápido
# Lista de columnas a eliminar
columnas_a_eliminar = [
    'Visa_Fvencimiento', 'Visa_mfinanciacion_limite', 'cliente_antiguedad',
    'Master_mfinanciacion_limite', 'Visa_mpagominimo', 'Visa_mpagospesos',
    'cproductos', 'chomebanking_transacciones', 'mtransferencias_recibidas',
    'Visa_mconsumospesos', 'Visa_cconsumos'
]

# Eliminar las columnas del DataFrame
data = data.drop(columns=columnas_a_eliminar)

# Ordenar el dataframe por cliente y mes
data= data.sort_values(by=['numero_de_cliente', 'foto_mes'])

# Ver los niveles de la variable 'foto_mes' 
print(data['foto_mes'].unique())

# Convertir 'foto_mes' a formato cadena
data['foto_mes'] = data['foto_mes'].astype(str)

# Filtrar las filas con 'foto_mes' a partir de enero 2021 (>= 202101)
data = data[data['foto_mes'] >= '202101']

# Excluir columnas específicas
exclude_cols = ['numero_de_cliente', 'foto_mes', 'clase_ternaria']

# Identificar las columnas para calcular Lag, dLag y percentiles
cols_to_process = [col for col in data.columns if col not in exclude_cols]

# Calulo percentiles
for col in cols_to_process:
      # Calcular Percentiles (sin incluir Lag ni dLag)
    data[f'percentile_{col}'] = data[col].rank(pct=True)

data.head()

print(data.columns)


# Guardo la base
# data.to_csv(r'C:\Users\tomas\Desktop\DM\DMEyF\datasets\data_Experimento.csv', index=False)



