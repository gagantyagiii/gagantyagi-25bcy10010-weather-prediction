# Importing modules to be used
import pandas as pd
import numpy as np
import streamlit as st
# Webpage content
st.write("# The Model is created using the dataset:")
df = pd.read_csv("Lucknow_1990_2022.csv") # Reading the dataset
st.write(df)
# Filling blank spaces using fillna method
df = df.fillna({'tmin':df['tmin'].mean(),
                'tmax':df['tmax'].mean(),
                'tavg':df['tavg'].mean(),
                'prcp':df['prcp'].mean(),
                'time':0
               })
# Properties of the dataframe
print(df.size)
print(df.shape)
print(df.columns)
print(df.prcp)
# Importing Scikit Learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# Test Train split
tavg_train, tavg_test = train_test_split(df['tavg'], test_size=0.2,random_state=42)
prcp_train, prcp_test = train_test_split(df['prcp'], test_size=0.2,random_state=42)
# Making data able to train the model
tavg_train_arr = np.array(tavg_train)
tavg_train_arr_fit = tavg_train_arr.reshape(-1,1)
tavg_test_arr = np.array(tavg_test)
tavg_test_arr_fit = tavg_test_arr.reshape(-1,1)
# Creating and training the model
model = LinearRegression()
model.fit(tavg_train_arr_fit,prcp_train)
# Taking predicted value output
prcp_predicted = model.predict(tavg_test_arr_fit)
# MSE
mse = mean_squared_error(prcp_test,prcp_predicted)
print(mse)
def value_prediction(a):
    '''This method returns predicted value of the model'''
    y = model.predict([[a]])
    return y[0]
    
# Webpage content
st.write("# Welcome to our weather prediction model.")
st.write("## Enter the average temperature:")
a = st.number_input("Average Temperature in Celsius")
if st.button("Predict"):
    st.write(f"The Predicted value is: {value_prediction(a)} mm")
