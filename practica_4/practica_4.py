#José María Flores San Martin - 1859565
#Graficas

from turtle import color
import pandas as pd
from tabulate import tabulate
from typing import Tuple, List
import matplotlib.pyplot as plt
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
        

df = pd.read_csv("cleaned_info.csv")


#primera grafica
plt.subplot(2, 1, 1)
#plt.figure(figsize=(8, 8))
#reviews de todos los juegos
x = df['yearpublished'].to_numpy()
y = df['average'].to_numpy()
scatter1 = plt.scatter(x, y, color='#3498DB', label='all boardgames', alpha=0.2, s=25)
scatXY = np.column_stack((x,y))

#calcular promedio de reviews por año
lista_year = list(set(scatXY[:,0]))
x = np.array(lista_year)
yearly = year_list(scatXY, lista_year)
#promedio anual de reviews
y = [est.mean(yearly[x]) for x in range(len(yearly))]

scatter2 = plt.scatter(x, y, color='#2874A6', label='boards year average')
plt.ylabel("Score", fontsize=14, labelpad=15)
plt.legend(loc='upper left')

plt.twinx()
y = stdDev(yearly)
plot1 = plt.plot(x,y, label = 'desviación')

plt.title("Average Year Score of Board Games Based On Years", fontsize=18, y=1.03)
plt.xlabel("Year", fontsize=14, labelpad=15)
plt.ylabel("Desviación", fontsize=14, labelpad=15)
plt.tick_params(labelsize=12, pad=6)

plt.legend(loc='upper right')

#segunda grafica
plt.subplot(2, 1, 2)
x = df['average'].to_numpy()
y = df['owned'].to_numpy()
plt.scatter(x,y, color='brown', label='Owned')
plt.xlabel("Score", fontsize=14, labelpad=15)
plt.ylabel("Owned", fontsize=14, labelpad=15)
plt.legend(loc='upper left')

plt.savefig('graficas_data_analysis.png', dpi=72)
plt.savefig('practica_3/graficas_data_analysis.png', dpi=72)
plt.show()