from numpy import genfromtxt
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
def get_input_data():
	x_train = genfromtxt('notMNIST_train_data.csv', delimiter=',')
	y_train = genfromtxt('notMNIST_train_labels.csv', delimiter=',')
	x_test = genfromtxt('notMNIST_test_data.csv', delimiter=',')
	y_test = genfromtxt('notMNIST_test_labels.csv', delimiter=',')
	return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)

def logistic():
	x_train,y_train,x_test,y_test=get_input_data()

	###Logistic Regression with L1 loss
	model_l1=LogisticRegression(C=0.00322,penalty='l1')
	model_l1.fit(x_train,y_train)
	print model_l1.score(x_test,y_test)
	
	###Logistic regression with L2 loss
	model_l2=LogisticRegression(C=0.000001,penalty='l2')
	model_l2.fit(x_train,y_train)
	print model_l2.score(x_test,y_test)

	coef_l1=model_l1.coef_.ravel()
	plt.imshow(np.abs(coef_l1.reshape(28, 28)),interpolation='nearest',cmap='binary')
	plt.show()
	coef_l2=model_l2.coef_.ravel()
	plt.imshow(np.abs(coef_l2.reshape(28, 28)),interpolation='nearest',cmap='binary')
	plt.show()
if __name__=='__main__':
	logistic()