#!/usr/bin/env python 3.6
# waloo le encoding: utf-8 de malade

"""
\033[32musage:	python mlp_predict.py [-s] dataset [weights_file]\033[0m

"""

#TODO

import sys
import csv
import os.path
import numpy as np
import neural_network
import matplotlib.pyplot as plt

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
	dataRaw = np.delete(dataRaw, [0,1,4,5], 1)

	# cast to float
	dataRaw = dataRaw.astype('float')

	# normalize data using transpose
	dataTemp = np.zeros((dataRaw.shape[1], dataRaw.shape[0]))
	for index, feature in enumerate(dataRaw.T):
		dataTemp[index] = [(x - min(feature)) / (max(feature) - min(feature)) for x in feature]
	
	print('\n\033[32mData loaded...\033[0m')
	print('\033[32m%d data rows for %d features...\033[0m\n' % (dataTemp.T.shape[0], dataTemp.T.shape[1]))
	return dataTemp.T, y

# load the weights and create the weighted mlp
def load_mlp(modelfile):
	if not os.path.isfile(modelfile):
		exit_error('can\'t find the file ' + modelfile)
	model = np.load(modelfile)
	if len(model) < 2:
		exit_error('file ' + modelfile + ' is not conform')
	# first item is model, following items are the weights & bias in layer order
	mlp = neural_network.net_loader(model[0])
	for i in range(1, len(model)):
		mlp[i-1].W = model[i][0]
		if mlp[i-1].bias == True:
			mlp[i-1].b = model[i][1]
	return mlp

def predict(csvfile, modelfile, param=0):

	# category to int (function for y)
	def f(i):
		if i == 1:
			return 'M'
		else:
			return 'B'

	params(param)

	# Data retrieval 
	mlp = load_mlp(modelfile)
	data, y = load_and_prep_data(csvfile)

	# Feed forward
	probas = neural_network.feed_forward(mlp, data, y, True)
	for i in range(len(probas)):
		error = 'ERROR' if f(np.argmax(probas[i])) != f(y[i]) else ''
		print('row {0:3d} [{1:.3f} {2:.3f}] => {3} {4} {5}'.format(i+1, probas[i][0], probas[i][1], \
			f(np.argmax(probas[i])), f(y[i]), error))

	# Results
	from sklearn.metrics import confusion_matrix, roc_auc_score
	tn, fp, fn, tp = confusion_matrix(np.argmax(probas, axis=1), y).ravel()
	print('\nConfusion matrix: ', confusion_matrix(np.argmax(probas, axis=1), y))
	print('Accuracy: {0:.4f}%'.format((tn+tp)/y.shape[0]))
	print('ROC AUC score: {0:.2f}'.format(roc_auc_score(y, np.argmax(probas, axis=1))))
	print('Cross entropy loss: {0:.4f}\n'.format(neural_network.cross_entropy_loss(probas, y)))

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
	if argc not in range(2, 5):
		print(__doc__)
	#traitement params
	param = 0 ; paramExists = False
	# if (sys.argv[1][0] == '-' and len(sys.argv[1]) == 2):
	# 	if sys.argv[1].find('v') > 0:
	# 		param += 1
	# 	else:
	# 		print(__doc__)
	#weightfile optional
	weightfile = './data/model.npy'
	if argc - param == 3:
		weightfile = sys.argv[-1]
	predict(sys.argv[1 - argc + param], weightfile, param)