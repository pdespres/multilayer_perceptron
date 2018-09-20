#!/usr/bin/env python 3.6
# waloo le encoding: utf-8 de malade

"""
\033[32musage:	python mlp_train.py [-b] [dataset]

Supported options:
	-b 		bonus		bonus fields\033[0m
"""

#TODO

#BONI
# exploration des donnees => features.py dataset_file

import sys
import csv
import os.path
import numpy as np
import neural_network

def load_and_prep_data(csvfile):

	# category to int function for y
	def f(i):
		if i[1] == 'M':
			return 1
		else:
			return 0

	#open file proc
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
	dataRaw = np.delete(dataRaw, [0,1,4,5], 1)

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
	limit = int(len(data) * train_share)
	p = np.random.permutation(len(data))
	data = data[p]
	y = y[p]
	print('\033[32mShuffling the dataset...\033[0m')
	return data[:limit], data[limit:], y[:limit], y[limit:]

def train(csvfile, param=0):
	params(param)
	# global parameters
	np.random.seed(42)
	train_share = 0.8			#share of the dataset (total=1) to use as train set
	mlp_layers = [10,20]		#size of each hidden layer
	mlp_init = ''				#random sur distrib 'uniform' or 'normal'(default normal)
	mlp_activation = ''			#'relu' (rectified linear unit) or 'sigmoid' or 'tanh'(hyperboloid tangent) (default tanh)
	nb_cats = 2					#size of the output layer
	epochs = 70
	batch_size = 128
	learningR = 0.01
	

	# Data retrieval and cleaning
	data, y = load_and_prep_data(csvfile)

	# Creation of train and validation dataset
	x_train, x_valid, y_train, y_valid = divide_dataset(data, y, train_share)
	batch_size = x_train.shape[0]
	print('\033[32m%d rows for the train dataset (%d%%), %d rows for validation...\033[0m\n' % \
		(x_train.shape[0], train_share * 100, x_valid.shape[0]))

	# Build Multilayer Perceptron according to parameters => neural_network.py
	mlp = neural_network.net_constructer(x_train.shape[1], nb_cats, mlp_layers, mlp_init, mlp_activation)
	print('\033[32mMultilayer Perceptron build...Hidden layers %s\033[0m\n' % (mlp_layers))
	
	for i in range(epochs):

		start = 0
		for j in range(round((x_train.shape[0] / batch_size) + .49)):

			end = min((j+1)*batch_size, x_train.shape[0])

			#feed forward
			probas = neural_network.feed_forward(mlp, x_train[start:end])
			# print(probas[:5])
			# #error mesure
			loss = neural_network.cross_entropy_loss(probas, y_train[start:end])
			print(i, loss)

			#back propagation
			neural_network.back_propagation(mlp, learningR, x_train[start:end], y_train[start:end])

			start = end

		# #print epoch info
		# if (i) % 1000 == 0:
		# 	print(i, loss)
		# 	probas = neural_network.feed_forward(mlp, x_train)
		# 	loss_t = neural_network.cross_entropy_loss(probas, y_train)
		# 	probas = neural_network.feed_forward(mlp, x_valid)
		# 	loss_v = neural_network.cross_entropy_loss(probas, y_valid)
		# 	print('epoch %d/%d - loss: %.4f - val_loss: %.4f' % ((i), epochs, loss_t, loss_v))

		#save epoch? ou save batch?

	#save model

	#Graph


def params(param):
	#load params according to the command line options
	params.bonus = False
	if param == 1:
		params.bonus = True
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
		if (sys.argv[1][0] == '-' and len(sys.argv[1]) == 2):
			if sys.argv[1].find('b') > 0:
				param += 1
			if param > 0:
				train(sys.argv[-1], param)
			else:
				print(__doc__)
		else:
			print(__doc__)
	else:
		train(sys.argv[-1])