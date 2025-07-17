import streamlit as st
import pandas as pd

st.title('License Management Software')
st.info('License Managament Software Thales')

with st.expander('Data'):
  st.write('Raw Data')
  df = pd.read_csv('synthetic_licensing_dataset.csv')
  st.dataframe(df)

