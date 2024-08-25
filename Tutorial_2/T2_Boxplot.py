# Tutorial 2: Example of Boxplot
import seaborn as sns
import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataPointsWOoutliers = [55, 57, 57, 58, 63, 66, 66, 67, 68, 69, 70, 70, 70, 70, 72, 73, 75, 76, 76, 78, 79, 81.]
dataPointsWoutliers  = [35, 57, 57, 58, 63, 66, 66, 67, 68, 69, 70, 70, 70, 70, 72, 73, 75, 76, 76, 78, 79, 99.]

df_combined = pd.DataFrame()
df_combined['normal'] = dataPointsWOoutliers
df_combined['outliers'] = dataPointsWoutliers

ax1 = sns.boxplot(data=df_combined, orient="v", palette="Set2")
ax1 = sns.swarmplot(data=df_combined, orient="v", color=(".25"))
plt.show()

print(statistics.median(dataPointsWOoutliers))
print(statistics.median(dataPointsWoutliers))