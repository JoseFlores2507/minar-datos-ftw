#José María Flores San Martin - 1859565
#Analisis Estadistico

from turtle import color
import pandas as pd
from tabulate import tabulate
from typing import Tuple, List
import numpy as np
import statistics as est


#funcion que genera una lista(años) de listas(reviews)
def year_list(scatxy, listaYear):
    total = [[] for y in range(len(listaYear))]
    for x in range(len(scatxy)):
        total[listaYear.index(scatxy[x,0])].append(scatxy[x,1])
    return total

#función de desviacion estandar
def stdDev(yearList):
    stdDevList = []
    for x in range(len(yearList)):
        if len(yearList[x]) == 1:
            stdDevList.append(1)
        else: stdDevList.append(est.stdev(yearly[x]))
    return stdDevList
        

df = pd.read_csv("csv/cleaned_info.csv")



year = df['yearpublished'].to_numpy()
avg = df['average'].to_numpy()
#usaremos esta lista de scatXY para calcular media y desviacion anual
scatXY = np.column_stack((year,avg))
#calcular promedio de reviews por año
lista_year = list(set(scatXY[:,0]))
yearly = year_list(scatXY, lista_year)

#calcula la media y desviacion de esta manera antes de conocer la funcion que use min_by_year
meanList = [est.mean(yearly[x]) for x in range(len(yearly))]
#desviacion estandar
SDList = stdDev(yearly)

min_by_year = df.groupby("yearpublished")\
    .aggregate(minYear=pd.NamedAgg(column="average", aggfunc=pd.DataFrame.min))
min_by_year.to_csv('csv/minYear.csv' ,index=True)

max_by_year = df.groupby("yearpublished")\
    .aggregate(maxYear=pd.NamedAgg(column="average", aggfunc=pd.DataFrame.max))
max_by_year.to_csv('csv/maxYear.csv' ,index=True)

boards_by_year = df.groupby("yearpublished").size()
boards_by_year.to_csv('csv/boardYear.csv' ,index=True)

