import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
df = pd.read_csv("government-expenditure-on-education.csv")
expenditureList = df ['recurrent_expenditure_total'].tolist()
yearList = df ['year'].tolist()
m_list = [[1]*len(yearList), yearList]
X = np.array(m_list).T
y = np.array(expenditureList)
w = inv(X.T @ X) @ X.T @ y
print(w)
y_line = X.dot(w)
plt.plot(yearList, expenditureList, 'o', label = 'Expenditure over the years')
plt.plot(yearList, y_line)
plt.xlabel('Year')
plt.ylabel('Expenditure')
plt.title('Education Expenditure')
plt.show()
y_predict = np.array([1, 2021]).dot(w)
print(y_predict)