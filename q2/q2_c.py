import sys,os
import numpy as np
import cPickle
import keras
from sklearn.metrics import accuracy_score 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras import backend as K
from sklearn import svm
def unpickle(file):
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict

def write_predictions(predictions):
	f=open('q2_c_output.txt','w')
	for prediction in predictions:
		f.write(label_names[prediction])
		f.write('\n')
	f.close()

def get_classes(inp_labels):
	y_labels=[]
	for row in inp_labels:
		for i in xrange(10):
			if row[i]==1:
				y_labels.append(i)
	return y_labels

if __name__ == "__main__":

	folder=sys.argv[1]
	test_file=sys.argv[2]

	batch = unpickle(os.path.join(folder,"batches.meta"))
	label_names = batch["label_names"]
	#print label_names

	a = unpickle(os.path.join(folder,"data_batch_1"))
	x_train = a["data"] 
	y_train = a["labels"]

	b = unpickle(os.path.join(folder,"data_batch_2"))
	x_train=np.vstack((x_train,b["data"]))
	y_train=np.vstack((y_train,b["labels"]))

	c = unpickle(os.path.join(folder,"data_batch_3"))
	x_train=np.vstack((x_train,c["data"]))
	y_train=np.vstack((y_train,c["labels"]))
	
	d = unpickle(os.path.join(folder,"data_batch_4"))
	x_train=np.vstack((x_train,d["data"]))
	y_train=np.vstack((y_train,d["labels"]))

	e = unpickle(os.path.join(folder,"data_batch_5"))
	x_train=np.vstack((x_train,e["data"]))
	y_train=np.vstack((y_train,e["labels"]))

	test = unpickle(test_file)
	x_test = test["data"] 
	#y_test = test["labels"]
	
	batch_size = 128
	num_classes = 10
	epochs = 15

	# input image dimensions
	img_rows, img_cols = 32, 32
	if K.image_data_format() == 'channels_first':
	    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
	    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
	    input_shape = (img_rows, img_cols, 3)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	#y_test = keras.utils.to_categorical(y_test, num_classes)

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	#model.add(Dense(250))
	#model.add(Dropout(0.25))	
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])

	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1)
	
	model2=Model(inputs=model.input,outputs=model.layers[12].output)
	train_weights=model2.predict(x_train)
	test_weights=model2.predict(x_test)

	#print train_weights.shape
	model_svm=svm.SVC(kernel='poly',degree=2,C=1,gamma=0.15)
	#print "yhaan"
	model_svm.fit(train_weights,get_classes(y_train))
	#print "whaan"
	predictions=model_svm.predict(test_weights)
	write_predictions(predictions)
	#print accuracy_score(predictions,get_classes(y_test))

	# score = model.evaluate(x_test, y_test, verbose=0)
	# print('Test loss:', score[0])
	# print('Test accuracy:', score[1])