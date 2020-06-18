import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib


import config

import preprocessors as pp

def run_training():
	"""Train the model."""

	data = pd.read_csv('titanic.csv')
	X_train,X_test,y_train,y_test = train_test_split(data.drop('survived',axis = 1),
													data['survived'],
													test_size = 0.2,
													random_state = 0)

	print(X_train.head())
	test_pipeline_1 = pp.CategoricalImputer(variables = config.CATEGORICAL_VARS)
	test_pipeline_2 = pp.NumericalImputer(variables = config.NUMERICAL_VARS_WITH_NA)
	test_pipeline_3 = pp.ExtractFirstLetter(variables = config.CABIN)
	test_pipeline_4 = pp.CategoricalEncoder(variables = config.CATEGORICAL_VARS)
	X_train = test_pipeline_1.fit_transform(X_train)
	X_train = test_pipeline_2.fit_transform(X_train)
	X_train = test_pipeline_3.fit_transform(X_train)
	print()
	print(X_train.head())
	print()
	X_train = test_pipeline_4.fit_transform(X_train)

	print(X_train.head())


if __name__ == '__main__':
	run_training()
