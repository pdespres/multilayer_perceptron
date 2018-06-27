#!/usr/bin/env python 3.6
# waloo le encoding: utf-8 de malade

"""
\033[32musage:	python mlp_train.py [-b] [dataset]

Supported options:
	-b 		bonus		bonus fields\033[0m
"""

#TODO

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
	
	return dataTemp.T, y

def divide_dataset(data, y, train_share):
	p = np.random.permutation(len(data))
	limit = int(len(data) * train_share)
	data = data[p]
	y = y[p]
	return data[:limit], data[limit:], y[:limit], y[limit:]

def train(csvfile, param=0):
	params(param)
	# global parameters
	np.random.seed(42)
	train_share = 0.8
	mlp_layers = [10,10]
	mlp_init = ['','']			#random sur distrib 'uniform' or 'normal'(default with '')
	mlp_activation = ['','']	#'relu' or 'sigmoid' or 'tanh'(default with '')

	# Data retrieval and cleaning
	data, y = load_and_prep_data(csvfile)
	x_train, x_valid, y_train, y_valid = divide_dataset(data, y, train_share)

	# Multilayer Perceptron construction according to parameters
	mlp = net_constructer(mlp_layers, mlp_init, mlp_activation)

	

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