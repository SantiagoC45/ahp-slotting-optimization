import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Optimización de slotting")

uploaded_file = st.file_uploader("Carga la base de datos de shipping y picking en un archivo excel con las hojas: `Shipping Detail Report` y `Labor Activity Report`", type=["xlsx", "xls"])

if uploaded_file:
    # Read uploaded Excel file into a DataFrame
    df = pd.read_excel(uploaded_file)
    st.write("Original Data")
    st.dataframe(df)