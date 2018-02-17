import numpy as np
import sys
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import accuracy_score,precision_recall_fscore_support

def get_input_data_train(filename):
	X = []; Y = []
	with open(filename, 'r') as f:
		for line in f:
			line = line.strip().split(',')
			Y.append(line[-1])
			X.append([float(x) for x in line[:len(line)-1]])
	X = np.asarray(X); Y = np.asarray(Y)
	return X,Y

def get_input_data_test(filename):
	X = []
	with open(filename, 'r') as f:
		for line in f:
			line = line.strip().split(',')
			X.append([float(x) for x in line])
	X = np.asarray(X)
	return X

def calculate_metrics(predictions,labels):
	accuracy=accuracy_score(labels, predictions)
	precision,recall,f1,y=precision_recall_fscore_support(predictions,labels,average='weighted')
	return accuracy,precision,recall

def regression(X_train,Y_train,X_test):
	ls=Ridge(alpha=0.0000001)
	ls.fit(X_train,Y_train)
	initial_predictions=ls.predict(X_test)
	predictions=[]
	for prediction in initial_predictions:
		if prediction<=0.5:
			predictions.append('0')
		else:
			predictions.append('1')
	#accuracy,precision,recall=calculate_metrics(predictions,Y_test)
	#print accuracy,precision,recall
	return predictions

if __name__=='__main__':
	train_file=sys.argv[1]
	test_file=sys.argv[2]
	X_train,Y_train=get_input_data_train(train_file)
	X_test=get_input_data_test(test_file)
	predictions=regression(X_train,Y_train,X_test)
	for prediction in predictions:
		print prediction