import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import random
import numpy as np


model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'reg_knn_cv.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

st.title("Concrete Data Insights")

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed/df_train.csv')
df = pd.read_csv(data_path)

df = df.drop(columns='compressive_strength')

fig, ax = plt.subplots(figsize=(8, 5))
df.hist(ax=ax)
plt.tight_layout()

st.pyplot(fig)

st.write('Create your own compressive strength prediction:')

random_values = {
    col: random.uniform(df[col].min(), df[col].max())
    for col in df.columns
}

prediction = pd.DataFrame([random_values])
 
water = st.slider('Water amount(kg/m2)', min_value=100, max_value=300)

prediction['water'] = water

pred = model.predict(prediction)

st.write('The compressive strength is:')
st.write(f'{pred} MPa')