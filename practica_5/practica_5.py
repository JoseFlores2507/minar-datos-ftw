#José María Flores San Martin - 1859565
#box plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

data = pd.read_csv("cleaned_info.csv")
df = pd.DataFrame(data)

df.boxplot("average", by="yearpublished", figsize=(15,9))
plt.xticks(rotation=45)
plt.title("Average por año", fontsize=16)
plt.xlabel("Años", fontsize=13)
plt.ylabel("Score", fontsize=13)

plt.savefig('boxplot_data.png', dpi=72)
plt.savefig('practica_5/boxplot_data.png', dpi=72)
plt.show()