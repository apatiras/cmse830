import streamlit as st
import pandas as pd
import seaborn as sns 

#reading the csv file 
wi_data=pd.read_csv("wi_data.csv")

st.write("Pick Features to display in pairplot")

radius_mean=st.checkbox("Radius_mean")
symmetry_worst=st.checkbox("Symmetry_worst")
area=st.checkbox("area_se")

s=[]

if radius_mean:
    s.append("radius_mean")

if symmetry_worst:
    s.append("symmetry_worst")

if area:
    s.append("area_se")


plot=sns.pairplot(wi_data[s])


g = sns.jointplot(
    data=wi_data,
    x="radius_mean", y="texture_mean", hue="diagnosis",
    kind="kde",
)

x=sns.catplot(
    data=wi_data, x="diagnosis", y="radius_mean",
    kind="violin", split=True, palette="pastel",
)

y=sns.relplot(data=wi_data, x="radius_mean", y="texture_mean",hue="diagnosis")


st.pyplot(plot)

st.write("M is positive for cancer and B is negative")
st.pyplot(g)
st.pyplot(x)
st.pyplot(y)

