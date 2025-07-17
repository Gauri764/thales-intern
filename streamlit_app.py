import streamlit as st

st.title('License Management Software')

st.write('Hello world!')

import pandas as pd

df = pd.read_csv('synthetic_licensing_dataset.csv')
st.dataframe(df)

