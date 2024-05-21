# -*- coding: utf-8 -*-
"""
Created on Sun May  5 23:27:27 2024

@author: ddiaz
"""

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
datos= pd.read_csv("FIFA2022.csv", sep=",")
datos2= pd.read_csv("FIFA - 2022.csv", sep=",")
encoder = OneHotEncoder()
da=datos["Team"]
categorical_columns = ['Team']

# Definir el preprocesamiento utilizando ColumnTransformer y Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), categorical_columns)
    ])

# Aplicar el preprocesamiento a los datos
processed_data = preprocessor.fit_transform(datos)
# Visualizar los datos procesados
print(processed_data.toarray())
print("")
danom= preprocessing.Normalizer().transform(datos2.T)
danom=danom.T
print(danom)
print("----------")
daes=preprocessing.StandardScaler().fit_transform(datos2)
print(daes)
print("-------------")
daes=preprocessing.RobustScaler().fit_transform(datos2)
print(daes)




