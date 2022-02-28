# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

from pandas import DataFrame
from pandas import concat

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#data time series transform to sliding windows

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# sigmoid

def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#tanh
def tanh(x, deriv=False):
    if (deriv==True):
        return (1-(x**2))
    return np.tanh(x)

#relu
    
def relu(x, derivative=False):
    if (derivative == True):
        for i in range(0, len(x)):
            for k in range(len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = 1
                else:
                    x[i][k] = 0
        return x
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            if x[i][k] > 0:
                pass  # do nothing since it would be effectively replacing x with x
            else:
                x[i][k] = 0
    return x

#parameter setting
    
errors =[]
epoch = 1000
lr = 0.1
hidden_dim = 4
feature = 2

iter_error = epoch/10
datax =[]

# input dataset
# =============================================================================
# file_list = ("HONN_1.csv", "HONN_2.csv")
# for file in file_list:
#     print(file)
#     df_inputx =  pd.read_csv(file, header=None)
#     datax = series_to_supervised(df_inputx,feature)
#     print "datax shape ", datax.shape
#     if file_list[0]==file:
#         data_all = datax
#     else:
#         data_all = data_all.append(datax)
#     
#     print "data all ", data_all.shape
# =============================================================================
df_input = pd.read_csv('HONN_1.csv', header=None)
#print df_input.shape

#data = series_to_supervised(df_input,feature)

#print data
data = df_input.values
print "shape of data ",data.shape
#print data[:,2]
X_temp = data[:,0:2]
y_temp = data[:,1:2]
X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=111)
X = X_train
y = y_train
print "X shape :", X.shape
#print "y shape ", y_temp.shape
#y = np.array([[0,0,1,1]]).T

print "y shape", y.shape
print "y temp shape", y_temp.shape
#random seed
#np.random.seed(999)



# initializaion
syn0 = 2*np.random.random((2,hidden_dim))-1
#b0 = 2*np.random.random((1,hidden_dim))-1
b0 = np.random.uniform(size=(1, hidden_dim))
syn1 = 2*np.random.random((hidden_dim,1))-1
#b1 = 2*np.random.random((1,1))-1
b1 = np.random.uniform(size=(1, 1))


for iter in xrange(epoch):
    
    # forward propagation
    l0 = X
    l01 = np.dot(l0,syn0)+ b0 # with bias
    l1 = relu(l01)   #with bias
    l02 = np.dot(l1,syn1)+b1 # with bias
    l2 = relu(l02)   #with bias
 #   l1 = tanh(np.dot(l0,syn0))
 #   l2 = tanh(np.dot(l1,syn1))
    
    # how much we missed?
    
    l2_error = y - l2
    
      # evaluate the error using the RMSE
       
    error = np.mean(np.sqrt((l2_error)**2))
    errors.append(error)
    if (iter% iter_error) == 0:
        print "Error after "+str(iter)+" iterations:" + str(error)
        #print "b0 ", b0
       # print "b1 ", b1
            
  
    #multiply how much we miss by the slope of derivative
    l2_delta = l2_error * relu(l2, True)
    
    l1_error = np.dot(l2_delta,syn1.T)
    if iter == 0:
        print "    l1 _error : ",l1_error.shape
        print "    syn1 : ", syn1.shape
        print "    l2 delta", l2_delta.shape
        print "    b0", b0
        print "    l01", l01.shape
        
    
    l1_delta = l1_error * relu(l1, True)
    #update weight
    
    syn1 += np.dot(l1.T, l2_delta)*lr
    b1 += np.sum(l2_delta, axis=0, keepdims=True)*lr    #with bias
    syn0 += np.dot(l0.T, l1_delta)*lr
    b0 += np.sum(l1_delta, axis=0, keepdims=True)*lr    #with bias
    
    
print "lr : ", lr
print "output After Training"
print l2.shape
print "RMSE Error After Training"
print error
print "R2 After Training", r2_score(y,l2)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(errors, alpha =0.4)
ax.set(title='error vs epoch', ylabel='error',xlabel='epoch')

plt.savefig('ann_jnp1.png')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
#ax1.plot(l2, alpha =0.4)
ax1.plot(y, alpha=0.5)
ax1.plot(l2)
ax1.set(title='predicted vs actual', ylabel='value',xlabel='data n')
plt.savefig('ann_jnp2.png')

plt.show()


# Test stage
X = X_test
y =  y_test
#print y.shape
#print l2.shape
# forward propagation
l0 = X
l1 = tanh(np.dot(l0,syn0))
l2 = tanh(np.dot(l1,syn1))
l2_error = y - l2
error = np.mean(np.sqrt((l2_error)**2))
    
result = np.column_stack((y,l2))
#print result
print "RMSE error test ", error
print "R2 After Training", r2_score(y,l2)    
