from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessors as pp
import config


titanic_pipe = Pipeline(
    [
    	('missing_indicator',
    		pp.MissingIndicator(variables = config.NUMERICAL_VARS_WITH_NA)),

    	('categorical_imputer',
    		pp.CategoricalImputer(variables = config.CATEGORICAL_VARS)),

    	('numerical_imputer',
    		pp.NumericalImputer(variables = config.NUMERICAL_VARS_WITH_NA)),

    	('extract_first_letter',
    		pp.ExtractFirstLetter(variables = config.CABIN)),

    	('rare_label_encoding',
    		pp.ExtractFirstLetter(variables = config.CATEGORICAL_VARS)),

    	('categorical_encoding',
    		pp.CategoricalEncoder(variables = config.CATEGORICAL_VARS)),

    	('scaler',
    		StandardScaler()),
    	('model',
    		LogisticRegression(C=0.0005, random_state=0))
    ]
    )