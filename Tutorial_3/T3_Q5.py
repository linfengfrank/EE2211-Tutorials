from scipy import stats

# define constants 
mu = 30 # mean = 30Ω
sigma = 1.8 # standard deviation = 1.8Ω
x1 = 28 # lower bound = 28Ω 
x2 = 33 # upper bound = 33Ω

## calculate probabilities
# probability from Z=0 to lower bound
p_lower = stats.norm.cdf(x1,mu,sigma) 
# probability from Z=0 to upper bound
p_upper = stats.norm.cdf(x2,mu,sigma)
# probability of the interval
Prob = (p_upper) - (p_lower)
# print the results print('\n')
print(f'Normal distribution: mean = {mu}, std dev = {sigma} \n')
print(f'Probability of occurring between {x1} and {x2}: ')
print(f'--> inside interval Pin = {round(Prob*100,1)}%') 
print(f'--> outside interval Pout = {round((1-Prob)*100,1)}% \n')
print('\n')