import pandas as pd
import numpy as np

from preprocessor import Pipeline
import config

import warnings
warnings.filterwarnings("ignore")



pipeline = Pipeline(target =  config.TARGET,
                    categorical_to_impute = config.CATEGORICAL_TO_IMPUTE,
                    year_variable = config.YEAR_VARIABLE,
                    numerical_to_impute = config.NUMERICAL_TO_IMPUTE,
                    numerical_log = config.NUMERICAL_LOG,
                    categorical_encode = config.CATEGORICAL_ENCODE,
                    features = config.FEATURES
                    )




if __name__ == '__main__':
    
    # load data set
    data = pd.read_csv(config.PATH_TO_DATASET)
    
    pipeline.fit(data)
    print('model performance')
    pipeline.evaluate_model()