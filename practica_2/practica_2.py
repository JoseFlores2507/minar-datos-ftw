#José María Flores San Martin
#Limpieza (Eliminar columnas y filas que no serviran)

from distutils.command.clean import clean
import requests
import io
import pandas as pd
from tabulate import tabulate
from typing import Tuple, List
import re
from datetime import datetime

#cargamos el csv a limpiar
df = pd.read_csv("games_detailed_info.csv")

#borrando columnas con valores NaN, 
#muchas es porque solo tienen como 500 datos 
# y 20K valores NaN, y otras son redundantes
df = df.dropna(axis=1)

#eliminar columna type porque todos los 
# elementos de la columna dicen 'boardgame'
df = df.drop('type', 1)

#borrando mas columnas que no serviran
#no tienen informacion importante
df = df.drop('Unnamed: 0', 1)
df = df.drop('suggested_num_players', 1)
df = df.drop('median', 1)
df = df.drop('numweights', 1)
df = df.drop('averageweight', 1)

#borrando filas acorde al año de publicación 
# al ser juegos de mesa tan viejos no tienen sentido que tenga reviews
df.drop(df[df['yearpublished'] < 1900].index, inplace = True)

is_multi = df["yearpublished"].value_counts() > 1
df = df[df["yearpublished"].isin(is_multi[is_multi].index)]
print(df)

df.to_csv('practica_2/cleaned_info.csv' ,index=False)
df.to_csv('cleaned_info.csv' ,index=False)