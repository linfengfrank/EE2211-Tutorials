# Question 5
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings 

warnings.filterwarnings("ignore",category=UserWarning)

df = load_iris()
Xdata = df['data']
Ydata = df['target']

acc_train_array = np.zeros(10)
acc_valid_array = np.zeros(10)

X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata,train_size=.8,test_size=.2)

validation_size = np.floor(len(y_train)/5)
train_size = len(y_train) - validation_size

for Nhidd in range (1,11,1):
    
    clf = MLPClassifier(hidden_layer_sizes=(Nhidd,Nhidd,Nhidd),solver='lbfgs', alpha=1e-5,random_state=1,max_iter=10000) 
    acc_train_array_fold = 0
    acc_valid_array_fold = 0
    for fold in range (1,6,1):
        # Construct validation and training matrices
        # int() to eliminate floating point portion, i.e. X.0 -> X
        Y_validation = y_train[(fold-1)*int(validation_size):int(validation_size)*fold]
        X_validation = X_train[(fold-1)*int(validation_size):int(validation_size)*fold,:]
        X_training = []
        Y_training = []
        Y_training = [y_train[Idx] for Idx in range (0,int(train_size),1) if ((Idx < (fold-1)*int(validation_size)) or (Idx >= int(validation_size)*fold)) ]        
        X_training = [X_train[Idx,:] for Idx in range (0,int(train_size),1) if ((Idx < (fold-1)*int(validation_size)) or (Idx >= int(validation_size)*fold)) ]
        Y_training = np.array(Y_training) # due to Y_training was in list format, need to convert them back to array        
        X_training = np.array(X_training) # due to X_training was in list format, need to convert them back to array        
        
        clf.fit(X_training,Y_training)
        ## trained output
        y_train_est = clf.predict(X_training)
        acc_train_array_fold += metrics.accuracy_score(y_train_est,Y_training)
        ## validation output
        y_valid_est = clf.predict(X_validation)
        acc_valid_array_fold += metrics.accuracy_score(y_valid_est,Y_validation)
        
    acc_train_array[Nhidd-1] = acc_train_array_fold/5
    acc_valid_array[Nhidd-1] = acc_valid_array_fold/5

## find the size that gives the best validation accuracy
Nhidden = np.argmax(acc_valid_array,axis=0)+1
print('Number of neuron per hidden layer with best validation accuracy is',Nhidden)
clf = MLPClassifier(hidden_layer_sizes=(Nhidden,Nhidden,Nhidden),solver='lbfgs', alpha=1e-5,random_state=1,max_iter=10000)
clf.fit(X_train,y_train)
y_train_est = clf.predict(X_train) 
y_test_est = clf.predict(X_test)
acc_train = metrics.accuracy_score(y_train_est,y_train)
acc_test = metrics.accuracy_score(y_test_est,y_test)
print('Test accuracy for Nhidd = ',Nhidden,' is ',acc_test*100,' %')

# plotting
hiddensize = [x for x in range(1,11)]
plt.plot(hiddensize, acc_train_array, color='blue', marker='o', linewidth=3, label='Training')
plt.plot(hiddensize, acc_valid_array, color='orange', marker='x', linewidth=3,    label='Validation')
plt.plot(Nhidden,acc_train, color='green', marker='o', markersize = 10, label ='Final Training')
plt.plot(Nhidden,acc_train, color='red', marker='X', markersize = 10,label = 'Test')
plt.xlabel('Number of hidden nodes in each layer')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracies')
plt.legend()
plt.show()
