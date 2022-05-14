import matplotlib.pyplot as plt
import statsmodels.api as sm
import numbers
import pandas as pd
from tabulate import tabulate




def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))

def transform_variable(df: pd.DataFrame, x:str)->pd.Series:
    if isinstance(df[x][0], numbers.Number):
        return df[x] # type: pd.Series
    else:
        return pd.Series([i for i in range(0, len(df[x]))])


def linear_regression(df: pd.DataFrame, x:str, y: str)->None:
    fixed_x = transform_variable(df, x)
    model= sm.OLS(df[y],sm.add_constant(fixed_x)).fit()
    print(model.summary())

    coef = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]['coef']
    df.plot(x=x,y=y, kind='scatter')
    plt.plot(df[x],[pd.DataFrame.mean(df[y]) for _ in fixed_x.items()], color='green')
    plt.plot(df_by_year[x],[ coef.values[1] * x + coef.values[0] for _, x in fixed_x.items()], color='red')
    plt.xticks(rotation=90)
    plt.savefig(f'img/lr_{y}_{x}.png')
    plt.show()


df = pd.read_csv("csv/cleaned_info.csv")

#print_tabulate(df.head(50))
df_by_year = df.groupby("yearpublished")\
    .aggregate(avgYear=pd.NamedAgg(column="average", aggfunc=pd.DataFrame.mean))
# df_by_year["sueldo_mensual"] = df_by_year["sueldo_mensual"]**10
df_by_year.reset_index(inplace=True)
print_tabulate(df_by_year.head(5))
linear_regression(df_by_year, "yearpublished", "avgYear")




"""year = data['yearpublished'].to_numpy()
avg = data['average'].to_numpy()
#usaremos esta lista de scatXY para calcular media y desviacion anual
scatXY = np.column_stack((year,avg))
#calcular promedio de reviews por año
lista_year = list(set(scatXY[:,0]))
yearly = year_list(scatXY, lista_year)
#medias
meanList = [est.mean(yearly[x]) for x in range(len(yearly))]

#avgYear = 
df = pd.DataFrame(np.column_stack((lista_year, meanList)))"""