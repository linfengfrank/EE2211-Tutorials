import pandas as pd; print("pandas version: {}".format(pd.__version__)) 
import matplotlib.pyplot as plt
import sklearn; print("scikit-learn version: {}".format(sklearn.__version__))
from sklearn.datasets import load_iris 

iris_dataset = load_iris()
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split( iris_dataset['data'], iris_dataset['target'], random_state=0)
# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names 
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# create a scatter matrix from the dataframe, color by y_train 
from pandas.plotting import scatter_matrix

grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20})
plt.show()


# method 2
import pandas as pd
import colorsys
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
iris = sns.load_dataset("iris")
g = sns.PairGrid(iris, hue="species")
g = g.map_diag(plt.hist, linewidth=3)
g = g.map_offdiag(plt.scatter)
plt.show()