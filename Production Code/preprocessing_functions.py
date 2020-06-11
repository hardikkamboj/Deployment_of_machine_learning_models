import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso

import joblib

def load_data(path):
	# function loads data from the given path
	return pd.read_csv(path)


def divide_train_test(df,target):
	# divides the dataset into trian and test
    X_train,X_test,y_train,y_test = train_test_split(df,
    	                                            df[target]
    	                                            ,test_size = 0.1,
    	                                            random_state = 0)
    return X_train,X_test,y_train,y_test



def impute_na(df,var,replacement):
	#replace the col of the df given with replacement
	return df[var].fillna(replacement)

def elapsed_years(df,var,ref_var):
	#reaplce the given col with difference with the reference var
	df[var] =  df[ref_var] - df[var]
	return df

def log_transform(df,var):
	#transfers the var to log of var
	return np.log(df[var])

def remove_rare_labels(df,var,freq_lables):
	#replace 'rare' for rare labels
	return np.where(df[var].isin(freq_lables),df[var],'Rare')	

def encode_categorical(df,var,mapping):
	#encodes the cateforical into numerical
	return df[var].map(mapping)


def train_scaler(df, output_path):
    scaler = MinMaxScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler


def scale_features(df, scaler):
    scaler = joblib.load(scaler) # with joblib probably
    return scaler.transform(df)


def train_model(df, target, output_path):
    # initialise the model
    lin_model = Lasso(alpha=0.005, random_state=0)
    
    # train the model
    lin_model.fit(df, target)
    
    # save the model
    joblib.dump(lin_model, output_path)
    
    return None


def predict(df, model):
    model = joblib.load(model)
    return model.predict(df)


  
    

