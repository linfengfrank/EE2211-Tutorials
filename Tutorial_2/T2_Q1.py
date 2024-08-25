import pandas as pd 
import matplotlib.pyplot as plt
import statistics

# Get the directory of the current script
# current_directory = os.path.dirname(os.path.abspath(__file__))

# Set this directory as the working directory
# os.chdir(current_directory)

df = pd.read_csv("./GovernmentExpenditureonEducation.csv")

expenditureList = df ['total_expenditure_on_education'].tolist() 
yearList = df ['year'].tolist()

plt.plot(yearList, expenditureList, label = 'Expenditure over the years')
plt.xlabel('Year') 
plt.ylabel('Expenditure') 
plt.title('Education Expenditure') 
plt.show()


# z-scrore normalization: long winded way
import numpy as np

averageExpenditureList = statistics.mean(expenditureList)
stdevExpenditureList = statistics.stdev(expenditureList)

subtraction = list(np.array(expenditureList) - np.array(averageExpenditureList))
normalizedExpenditureList = list(np.array(subtraction)/np.array(stdevExpenditureList))
print(normalizedExpenditureList)