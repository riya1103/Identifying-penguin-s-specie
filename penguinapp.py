import pandas as pd
import numpy as np
import pickle
import streamlit as st
st.subheader("Penguin Prediction App")
st.subheader("Find your pinguu type:)")
df = pd.read_csv("penguins_cleaned.csv")
df.info()

df['island'].nunique()

df['species'].nunique()
df['island'].value_counts()
df['species'].value_counts()
df.columns

#We fill min, max and mean values for the slider
df.describe()


st.sidebar.header("User Input Features")

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox("Island",("Biscoe","Dream","Torgersen"))
        sex = st.sidebar.selectbox("Sex",("male","female"))
        bill_length_mm = st.sidebar.slider("bill_length_mm",32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider("bill_depth_mm",13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider("flipper_length_mm",172.0,231.0,200.9)
        body_mass_g = st.sidebar.slider("body_mass_g",2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        df2 = pd.DataFrame(data,index=[0])
        return df2
    input_df = user_input_features()

X = df.drop(columns=['species'])
y = df['species']
X1 = pd.concat([input_df,X],axis = 0)
X1 = pd.get_dummies(X1)
X =pd.get_dummies(X)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X,y)

# Apply model to make predictions
prediction = clf.predict(X1[:1])
prediction_proba = clf.predict_proba(X1[:1])
st.write("This web app predicts the species of penguins as a function of their input parameters (bill length, bill width, flipper length, body mass, sex and island.")
st.write("The predicted specie is:")
st.write(prediction)
st.subheader('Prediction Probability for the different species')
st.write("0:Adelie",
         "1:Chinstrap",
         "2:Gentoo")
st.write(prediction_proba)
st.write("Created by: Riya Jain")





