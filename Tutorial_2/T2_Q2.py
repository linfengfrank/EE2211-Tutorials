import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("AnnualMotorVehiclePopulationbyVehicleType.csv")
year = df ['year'].tolist() 
category = df ['category'].tolist() 
vehtype = df ['type'].tolist() 
number = df ['number'].tolist() 

val1 = df.loc[df['type']=='Omnibuses'].index 
val2 = df.loc[df['type']=='Excursion buses'].index 
val3 = df.loc[df['type']=='Private buses'].index 
print(val1)
print(val2)
print(val3)

#print(df.loc[df['type']=='Omnibuses'])

List1 = df.loc[val1]; print(List1)
List2 = df.loc[val2]; print(List2)
List3 = df.loc[val3]; print(List3)

plt.plot(List1['year'], List1['number'], label = 'Number of Omnibuses')
plt.plot(List2['year'], List2['number'], label = 'Number of Excursion buses')
plt.plot(List3['year'], List3['number'], label = 'Number of Private buses')
plt.xlabel('Year')
plt.ylabel('Number of vehicles')
#plt.xticks(List1['year'])
plt.title('Number of vehicles over the years') 
plt.legend()
plt.show()


# method 2, using seaborn
import pandas as pd
import colorsys
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("AnnualMotorVehiclePopulationbyVehicleType.csv")

sns.set_style("darkgrid")

df3 = df.loc[df['type'].isin(['Omnibuses', 'Excursion buses', 'Private buses'])]

g = sns.PairGrid(data=df3, x_vars="year", y_vars="number", hue="type", height=10, aspect=1)
g = g.map(plt.plot, alpha=0.5)
g = g.set(xlim=(df['year'].min(), df['year'].max()))
g = g.add_legend()
g.fig.suptitle('Number of vehicles over the years')
plt.xlabel('Year')
plt.ylabel('Number of vehicles')

plt.show()