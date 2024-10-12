#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:45:29 2020

@author: thomas
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def Tut8_Q2():
    
    # load data
    df = pd.read_csv("government-expenditure-on-education.csv")
    expenditure = df['total_expenditure_on_education'].to_numpy()
    years = df['year'].to_numpy()

    # create normalized variables
    max_expenditure = max(expenditure)
    max_year = max(years)
    y = expenditure/max_expenditure
    X = np.ones([len(y), 2])
    X[:, 1] = years/max_year

    # Gradient descent 
    learning_rate = 0.1
    w = np.zeros(2)
    pred_y, cost, gradient = exp_cost_gradient(X, w, y)
    num_iters = 2000000
    cost_vec = np.zeros(num_iters)
    print('Initial Cost =', cost)
    for i in range(0, num_iters):
        
        # update w
        w = w - learning_rate*gradient
        
        # compute updated cost and new gradient
        pred_y, cost, gradient = exp_cost_gradient(X, w, y)
        cost_vec[i] = cost
        
        if(i % 200000 == 0):            
            print('Iter', i, ': cost =', cost)
        
    pred_y, cost, gradient = exp_cost_gradient(X, w, y)
    print('Final Cost =', cost)
    
    # Plot cost function values over iterations
    plt.figure(0, figsize=[9,4.5])
    plt.rcParams.update({'font.size': 16})
    plt.plot(np.arange(0, num_iters, 1), cost_vec)
    plt.xlabel('Iteration Number') 
    plt.ylabel('Square Error')
    plt.xticks(np.arange(0, num_iters+1, 500000))
    plt.title('Learning rate = ' + str(learning_rate))
    plt.savefig('FigTut8Cost' + str(learning_rate) + '.png')
    
    # Extrapolate until year 2023 
    ext_years = np.arange(min(years), 2024, 1)
    ext_X = np.ones([len(ext_years), 2])
    ext_X[:, 1] = ext_years/max_year
    pred_y = np.exp(-ext_X @ w)
    
    # Plot extrapolation
    plt.figure(1, figsize=[9,4.5])
    plt.rcParams.update({'font.size': 16})
    plt.scatter(years, expenditure, s=20, marker='o', c='blue', label='real data')
    plt.plot(ext_years, pred_y * max_expenditure, c='red', label='fitted curve')
    plt.xlabel('Year') 
    plt.ylabel('Expenditure')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.legend(loc='upper left',ncol=3, fontsize=15)
    plt.savefig('FigTut8Extrapolation' + str(learning_rate) + '.png')

    plt.show()
   



def exp_cost_gradient(X, w, y):
    
    # Compute prediction, cost and gradient based on mean square error loss
    pred_y = np.exp(-X @ w)
    cost   = np.sum((pred_y - y)*(pred_y - y)) 
    gradient = -2 * (pred_y - y) * pred_y @ X 
    
    return pred_y, cost, gradient

if __name__ == '__main__':
    Tut8_Q2() 