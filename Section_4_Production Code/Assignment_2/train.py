import preprocessing_functions as pf
import config

import warnings
warnings.simplefilter(action='ignore')
# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
data = pf.load_data(config.PATH_TO_DATASET)
# divide data set

X_train,X_test,y_train,y_test = pf.divide_train_test(data,'survived')

# get first letter from cabin variable

X_train['cabin'] = pf.extract_cabin_letter(X_train,'cabin')

# impute categorical variables

for var in config.CATEGORICAL_VARS:
	X_train[var] = pf.impute_na(X_train,var)

# impute numerical variable

for var in config.NUMERICAL_TO_IMPUTE:
	X_train = pf.add_missing_indicator(X_train,var)
	X_train[var] = pf.impute_na(X_train,var,value = config.IMPUTATION_DICT[var])

# Group rare labels

for col in config.CATEGORICAL_VARS:
	X_train[col] = pf.remove_rare_labels(X_train,col,freq_labels = config.FREQUENT_LABELS[col])

# encode categorical variables

oh = pf.train_encoder(X_train,config.CATEGORICAL_VARS,config.OUTPUT_ENCODER_PATH)
X_train = pf.encode_categorical(X_train,config.CATEGORICAL_VARS,config.OUTPUT_ENCODER_PATH)
print(X_train.shape)
print(X_train.head())
# check all dummies were added

X_train = pf.check_dummy_variables(X_train,config.DUMMY_VARIABLES)

# train scaler and save

scaler = pf.train_scaler(X_train,config.OUTPUT_SCALER_PATH)

# scale train set

X_train = scaler.transform(X_train)
# train model and save

pf.train_model(X_train,y_train,config.OUTPUT_MODEL_PATH)

print('Finished training')