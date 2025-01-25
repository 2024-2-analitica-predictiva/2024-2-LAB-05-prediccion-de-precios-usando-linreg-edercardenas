#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import os
import json
import gzip
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error


# Paso 1.
print('Paso 1...')
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
df_train = pd.read_csv('files/input/train_data.csv.zip', index_col=False, compression="zip")
df_test = pd.read_csv('files/input/test_data.csv.zip', index_col=False, compression="zip")

df_train['Age'] = 2021 - df_train['Year']
df_test['Age'] = 2021 - df_test['Year']

df_train.drop(columns=['Year', 'Car_Name'], inplace=True)
df_test.drop(columns=['Year', 'Car_Name'], inplace=True)

# Eliminamos los registros con informacion no disponible
df_train = df_train.dropna()
df_test = df_test.dropna()

# # Eliminamos los registros duplicados
# df_train = df_train.drop_duplicates()
# df_test = df_test.drop_duplicates()

# Paso 2.
print('Paso 2...')
# Divida los datasets en x_train, y_train, x_test, y_test.
x_train = df_train.drop(columns=['Present_Price'])
y_train = df_train['Present_Price']

x_test = df_test.drop(columns=['Present_Price'])
y_test = df_test['Present_Price']


# Paso 3.
print('Paso 3...')
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.


# Creamos el transformer
transformer = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(), ['Fuel_Type', 'Selling_type', 'Transmission']),
        ('scaler', MinMaxScaler(), ['Selling_Price', 'Driven_kms', 'Owner', 'Age']),
    ],
    # remainder="passthrough",
)

# Creamos el pipeline
pipeline = Pipeline(
    steps =[
        ('transformer', transformer),
        ('feature_selection', SelectKBest(score_func=f_regression)),
        # ('scaler2', MinMaxScaler()),
        ('linearregression', LinearRegression()),
    ],
    verbose=False,
)

# Paso 4.
print('Paso 4...')
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
params = {
    'feature_selection__k': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

grid = GridSearchCV(pipeline, params, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1, refit=True)

grid.fit(x_train, y_train)

print('Mejores hiperparametros:', grid.best_params_)
print('score_test:', grid.score(x_test, y_test))


# Paso 5.
print('Paso 5...')
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
if not os.path.exists("files/models"):
        os.makedirs("files/models")

with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid, f)


# Paso 6.
print('Paso 6...')
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json.
y_train_pred = grid.predict(x_train)
y_test_pred = grid.predict(x_test)

metrics = {
    'type': 'metrics',
    'dataset': 'train',
    'r2': r2_score(y_train, y_train_pred),
    'mse': mean_squared_error(y_train, y_train_pred),
    'mad': median_absolute_error(y_train, y_train_pred)
}

if not os.path.exists("files/output"):
        os.makedirs("files/output")

with open("files/output/metrics.json", "w") as f:
    json.dump(metrics, f)


metrics = {
    'type': 'metrics',
    'dataset': 'test',
    'r2': r2_score(y_test, y_test_pred),
    'mse': mean_squared_error(y_test, y_test_pred),
    'mad': median_absolute_error(y_test, y_test_pred)
}

with open("files/output/metrics.json", "a") as f:
    f.write('\n' + json.dumps(metrics))