import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler , LabelEncoder , OneHotEncoder
import pandas as pd
import pickle

# Loading our trained model

model = tf.keras.models.load_model('model.h5')

# Loading our encoders and scaler

with open('label_encode_gender.pkl' , 'rb') as file:
    label_encode_gender = pickle.load(file)

with open('one_hot_geo.pkl' , 'rb') as file:
    one_hot_geo = pickle.load(file)

with open('scaler.pkl' , 'rb') as file:
    scaler = pickle.load(file)

## Streamlit 

st.title("Customer Churn Prediction")

# User Input

geography = st.selectbox('Geography' , one_hot_geo.categories_[0])
gen = st.selectbox('Gender' , label_encode_gender.classes_)
age = st.slider('Age' , 18 , 90)
balance = st.number_input('Balance')
credit_score = st.number_input('CreditScore')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure' , 0 , 10)
num_of_products = st.slider('NumOfProducts' , 1 , 4)
has_cr_card = st.selectbox('HasCrCard' , [0,1])
is_active_member = st.selectbox('IsActiveMember' , [0,1])

# Prepare the Input Data

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encode_gender.transform([gen])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

# Converting above data into dataframe
input_df = pd.DataFrame(input_data)

# OneHot encode of geography

geo_encoded = one_hot_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded , columns = one_hot_geo.get_feature_names_out(['Geography']))

# Combining with input_df
input_data = pd.concat([input_data.reset_index(drop = True) , geo_encoded_df] , axis = 1)

# Scale the input df
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)
pred_prob = prediction[0][0]

st.write('Churn Probability : ',pred_prob)

if pred_prob > 0.5 :
    st.write('The Customer is likely to Churn....')

else:
    st.write('Customer is Loyal....')

