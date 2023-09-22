import plotly.express as px
import streamlit as st
import seaborn as sns

# Load the Iris dataset from Seaborn
df = sns.load_dataset('iris')

# Create a 3D scatter plot using Plotly
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width', color='species')

# Streamlit app
st.title("Iris Dataset 3D Scatter Plot")
st.write("This interactive 3D scatter plot visualizes the Iris dataset with Petal Width, Petal Length, and Sepal Length as the axes.")

st.plotly_chart(fig, use_container_width=True)
