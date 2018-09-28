#!/usr/bin/env python 3.6
# waloo le encoding: utf-8 de malade

"""
\033[32musage:	python mlp_train.py [-rk] [dataset]

Supported options:
	-r 		regul			L2 regularization to avoid over-fitting
	-k		keep graph 		re-use the previous graph
	-a 		adam 			optimization with adam method\033[0m
"""

#TODO
# droupout layer?
#BONI
# exploration des donnees => features.py dataset_file
# L2 regul
# possible de garder le meme graph

import sys
import csv
import os.path
import numpy as np
import neural_network
import matplotlib.pyplot as plt
import pickle

def load_and_prep_data(csvfile):

	# category to int function for y
	def f(i):
		if i[1] == 'M':
			return 1
		else:
			return 0

	# open file proc
	def load_data(csvfile):
		if not os.path.isfile(csvfile):
			exit_error('can\'t find the file ' + csvfile)
		data = []
		with open(csvfile) as csv_iterator:
			data_reader = csv.reader(csv_iterator, delimiter=',')
			for row in data_reader:
				data.append(row)
		csv_iterator.close()
		if len(data) < 1:
			exit_error('file ' + csvfile + ' is empty')
		return data

	# load data from csvfile
	dataRaw = np.array(load_data(csvfile))
	dataTemp = []

	# fill y / replace categorical values with numeric values (1 is for 'M')
	y = np.array([f(i) for i in dataRaw])

	# remove unwanted columns/features
	# dataRaw = np.delete(dataRaw, [0,1,4,5], 1)
	dataRaw = np.delete(dataRaw, [0,1], 1)

	# cast to float
	dataRaw = dataRaw.astype('float')

	# normalize data using transpose
	dataTemp = np.zeros((dataRaw.shape[1], dataRaw.shape[0]))
	for index, feature in enumerate(dataRaw.T):
		dataTemp[index] = [(x - min(feature)) / (max(feature) - min(feature)) for x in feature]
	
	print('\n\033[32mData loaded...\033[0m')
	print('\033[32m%d data rows for %d features...\033[0m' % (dataTemp.T.shape[0], dataTemp.T.shape[1]))
	return dataTemp.T, y

def divide_dataset(data, y, train_share):
	# shuffle & divide in 2 parts according to train_share & divide X and Y
	limit = int(len(data) * train_share)
	p = np.random.permutation(len(data))
	data = data[p]
	y = y[p]
	print('\033[32mShuffling the dataset...\033[0m')
	return data[:limit], data[limit:], y[:limit], y[limit:]

def draw_graph(errors_t, errors_v, epochs):
	graph = './data/graph.pickle'
	param_range = np.arange(1, epochs+1, 1)
	color = np.random.random(3)
	if os.path.isfile(graph) and params.keep:
		with open(graph, 'rb') as pickle_file:
			ax = pickle.load(pickle_file)
	else:
		ax = plt.subplot(111)
		plt.title("Validation Curve")
		plt.xlabel("Epochs")
		plt.ylabel("Cross entropy loss")
		plt.tight_layout()
		plt.legend(loc="best")
	# plt.plot(param_range, errors_t, label="Training score", color="black")
	# plt.plot(param_range, errors_v, label="Validation score", color="green")
	plt.plot(param_range, errors_t, '--', label="Training score", color=color)
	plt.plot(param_range, errors_v, label="Validation score", color=color)
	if os.path.isfile(graph):
		os.remove(graph)
	pickle.dump(ax, open(graph, 'wb'))
	plt.show()

def gradient_descent():
	return

def train(csvfile, param=0):
	params(param)
	# global parameters
	np.random.seed(42)
	train_share = 0.7			#share of the dataset (total=1) to use as train set
	mlp_layers = [100,100]		#size of each hidden layer
	mlp_init = ''				#random sur distrib 'uniform' or 'normal'(default normal)
	mlp_activation = ''			#'relu' (rectified linear unit) or 'sigmoid' or 'tanh'(hyperboloid tangent) (default relu)
	nb_cats = 2					#size of the output layer
	epochs = 3000					#number of times the whole dataset will be read
	batch_size = 128			#for adam: number of data rows that will be processed together
	learningR = 0.05			#modifier applied to weights update
	regL2 = 0.03
	es_nb = 2					#for early stopping: number of times the error has to go up before stopping

	# Data retrieval and cleaning
	data, y = load_and_prep_data(csvfile)

	# Creation of train and validation dataset
	x_train, x_valid, y_train, y_valid = divide_dataset(data, y, train_share)

	# Mod to run without boni
	if not params.adam:
		batch_size = x_train.shape[0]
	if not params.regul:
		regL2 = 0.

	print('\033[32m%d rows for the train dataset (%d%%), %d rows for validation...\033[0m\n' % \
		(x_train.shape[0], train_share * 100, x_valid.shape[0]))

	# Build Multilayer Perceptron according to parameters => refer to neural_network.py
	mlp = neural_network.net_constructer(x_train.shape[1], nb_cats, mlp_layers, mlp_init, mlp_activation)
	print('\033[32mMultilayer Perceptron build...Hidden layers %s\033[0m\n' % (mlp_layers))
	
	errors_v = [] ; errors_t = [] ; es_model = [] ; test = []
	early_stopping = False ; es_cpt = 0
	for i in range(epochs):

		if early_stopping:
			#rollback restore best model weights and bias
			for l in range(len(mlp)):
				# print(np.array_equal(mlp[l].W, es_model[l][0]))
				# print(np.array_equal(test, es_model[l][0]))
				mlp[l].W[:] = es_model[l][0]
				mlp[l].b[:] = es_model[l][1]
			break
		# if params.adam:
		# 	return
		# else:
		# 	gradient_descent()

		start = 0
		for j in range(round((x_train.shape[0] / batch_size) + .49)):

			end = min((j+1)*batch_size, x_train.shape[0])

			#feed forward
			probas = neural_network.feed_forward(mlp, x_train[start:end], y_train[start:end])

			#back propagation
			neural_network.back_propagation(mlp, learningR, regL2, x_train[start:end])

			start = end

		# print epoch info
		if (i+1) % 100 == (i+1) % 100:
		# if (i+1) % 100 == 0:
			probas = neural_network.feed_forward(mlp, x_train, y_train)
			loss_t = neural_network.cross_entropy_loss(probas, y_train)
			probas = neural_network.feed_forward(mlp, x_valid, y_valid)
			loss_v = neural_network.cross_entropy_loss(probas, y_valid)
			print('epoch %d/%d - loss: %.4f - val_loss: %.4f' % ((i+1), epochs, loss_t, loss_v))
			if params.es and len(errors_v) > 0 and loss_v >= np.max(errors_v[-1]):
				es_cpt += 1
				if es_cpt == es_nb:
					early_stopping = True
			else:
				es_cpt = 0
				# Save model weights and bias
				es_model = [] 
				for k in range(len(mlp)):
					temp = [] ; layer = []
					temp[:] = mlp[k].W
					layer.append(temp)
					temp[:] = mlp[k].b
					layer.append(temp)
					# layer[:] = [mlp[k].W, mlp[k].b]
					es_model.append(layer)
			errors_v.append(loss_v) ; errors_t.append(loss_t)

		#save epoch? ou save batch?

		# shuffle dataset for next epoch
		p = np.random.permutation(len(x_train))
		x_train = x_train[p]
		y_train = y_train[p]

	# for dev
	from sklearn.metrics import confusion_matrix
	tn, fp, fn, tp = confusion_matrix(np.argmax(probas, axis=1), y_valid).ravel()
	print(confusion_matrix(np.argmax(probas, axis=1), y_valid))
	print('accuracy: ', (tn+tp)/y_valid.shape[0])
	probas = neural_network.feed_forward(mlp, data, y)
	tn, fp, fn, tp = confusion_matrix(np.argmax(probas, axis=1), y).ravel()
	print(confusion_matrix(np.argmax(probas, axis=1), y))
	print('accuracy: ', (tn+tp)/y.shape[0])
	probas = neural_network.feed_forward(mlp, x_valid, y_valid)
	print('val_loss: ', neural_network.cross_entropy_loss(probas, y_valid))

	#Graph
	draw_graph(errors_t, errors_v, i + 1 - early_stopping)

	#save model
	model = []
	model.append(neural_network.topology(mlp))
	for i in range(len(mlp)):
		layer = [mlp[i].W, mlp[i].b]
		model.append(layer)
	np.save('./data/model.npy', model)

def params(param):
	#load params according to the command line options
	params.regul = False
	params.keep = False
	params.adam = False
	params.es = True
	if param in (1,3,5):
		params.regul = True
	if param in (2,3):
		params.keep = True
	if param in (4,5):
		params.adam = True
	return

def exit_error(string):
	print(string)
	sys.exit(42)
	return

if __name__ == "__main__":
	argc = len(sys.argv)
	if argc not in range(2, 4):
		print(__doc__)
	elif argc == 3:
		#traitement params
		param = 0
		count = 0
		if (sys.argv[1][0] == '-' and len(sys.argv[1]) in range(2,4)):
			if sys.argv[1].find('r') > 0:
				param += 1
				count += 1
			if sys.argv[1].find('k') > 0:
				param += 2
				count += 1
			if sys.argv[1].find('a') > 0:
				param += 4
				count += 1
			# if param > 0 and param < 6 and (count + 1) == len(sys.argv[1]):
			if param > 0 and (count + 1) == len(sys.argv[1]):
				train(sys.argv[-1], param)
			else:
				print(__doc__)
		else:
			print(__doc__)
	else:
		train(sys.argv[-1])