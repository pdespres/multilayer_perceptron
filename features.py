#!/usr/bin/env python 3.6
# waloo le encoding: utf-8 de malade

"""
\033[32musage:	python features.py dataset_file.csv\033[0m

"""

import sys
import pandas as pd
from sklearn import decomposition
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import os.path

# https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names

def features(csvfile):
	if csvfile == '':
		csvfile = './data/data.csv'
	if not os.path.isfile(csvfile):
		exit_error('can\'t find the file ' + csvfile)
	df = pd.read_csv('./data/data.csv', header=None)
	df[1] = df[1].astype('category').cat.codes
	# column 0 looks like an index. transfo to char length to see if it's useful
	df[0] = [len(str(x)) for x in df[0]]

	for i in range(7):
		print('\n', df.iloc[:,(i*5):(i+1)*5].describe(include='all'))
	print('Malignant: ', len(df[df[1]==1]))
	print('Benign: ', len(df[df[1]==0]))

	plt.figure(figsize=(22,15))
	plt.subplots_adjust(bottom=None, top=0.95)

	# Correlation matrix
	ax = plt.subplot(2,1,1)
	corr_matrix = df.corr()
	ax = sns.heatmap(corr_matrix, ax=ax, annot=True, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)
	plt.title('Correlation matrix', size = 12)

	# 2D drawing after dimensionality reduction
	plt.subplot(2,1,2)
	y = df[1]
	df.drop([0,1,2,5], axis=1, inplace=True)
	df = (df-df.mean())/df.std()

	pca = decomposition.PCA(n_components=2)
	pca.fit(df)
	train_red = pca.transform(df)
	plt.scatter(train_red[:,0],train_red[:,1], c=y, alpha=0.4)
	plt.title('Data visualization atfer 2D PCA', size = 12)

	plt.show()

def exit_error(string):
	print(string)
	sys.exit(42)
	return

if __name__ == "__main__":
	features('')