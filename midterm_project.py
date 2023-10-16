import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import hiplot as hip
import json
import numpy as np
import plotly.figure_factory as ff

data = pd.read_csv("nigeria_houses_data.csv")


#introductory material 
st.markdown(" # Welcome! Explore Nigeria's Housing Market")

st.markdown("The purpose of this app is to help users explore the housing market in various parts of Nigeria. Please enjoy!")

image = Image.open("project_photo.jpg")
st.image(image, caption="Lagos Island skyline from Victoria Island")

#interactive chart 
# 3D scatter plot using Plotly
fig = px.scatter_3d(data, x='bedrooms', y='bathrooms', z='parking_space', color='state')
st.title("Nigeria Housing Dataset 3D Scatter Plot")
st.write("This interactive 3D scatter plot visualizes the housing dataset with bedrooms, bathrooms, and parking as the axes.")
st.plotly_chart(fig, use_container_width=True)

###############################################################################
#Exploratory Data Analysis
#
#Hiplot Visualization 
st.header("Hiplot visualization")
exp = hip.Experiment.from_dataframe(data)
st_hiplot=st.empty()
st.write(exp)

#################################################################
#Correlation heatmap
fig, ax = plt.subplots()
sns.heatmap(data.corr(), annot= True, ax=ax)
st.write(fig)

#############################################################################
#relplot
data_columns = list(data.columns.values)
option1 = st.selectbox('Choose a x-value',(data_columns))
option2 = st.selectbox('Choose a y-value',(data_columns))
st.write('You selected:', option1, "and", option2)
st.pyplot(sns.relplot(data=data, x= option1,y=option2))
##############################################################################
st.title("Average Housing Price by State")

# Create a dropdown to select the state
selected_state = st.selectbox("Select a State", data['state'].unique())

# Input for the exchange rate
exchange_rate = st.number_input("Enter the Exchange Rate (NGN to USD)", value=0.0013)

# Filter the data by the selected state
filtered_data = data[data['state'] == selected_state]

# Calculate the average price for the selected state
avg_price_ngn = filtered_data['price'].mean()
avg_price_usd = avg_price_ngn * exchange_rate

# Display the average price in both NGN and USD
st.write(f"Average Price of Housing in {selected_state}:")
st.write(f"In NGN: ₦{avg_price_ngn:.2f}")
st.write(f"In USD: ${avg_price_usd:.2f}")
average_prices_by_state = data.groupby('state')['price'].mean().reset_index()
average_prices_usd = average_prices_by_state['price'] * exchange_rate
average_prices_by_state['price_usd'] = average_prices_usd
plt.bar(average_prices_by_state['state'], average_prices_by_state['price_usd'])
plt.xlabel('State')
plt.ylabel('Average Price (USD)')
plt.title('Average Housing Price by State (USD)')
st.pyplot()

# Optionally, you can display the data table for the selected state
st.write("Data for", selected_state)
st.write(filtered_data)


#######################################################################
#interactive histogram 
st.title("Housing Data Analysis")
characteristic = st.selectbox("Select a Housing Characteristic", ['bathrooms', 'bedrooms', 'toilets', 'parking_space', 'state'])

filtered_data = data[characteristic].dropna()  # Drop any NaN values

fig = ff.create_distplot([filtered_data], [characteristic], bin_size=[0.1])
st.plotly_chart(fig, use_container_width=True)
st.write(f"Summary statistics for {characteristic}:")
st.write(filtered_data.describe())


st.write("Pick Features to display in pairplot")

bathrooms=st.checkbox("bathrooms")
bedrooms=st.checkbox("bedrooms")
toilets=st.checkbox("toilets")
parking_space=st.checkbox('parking_space')

###########################################################################
#relplot
data = pd.read_csv("nigeria_houses_data.csv")

s=[]
if "bathrooms" in data.columns:
    s.append("bathrooms")
if "bedrooms" in data.columns:
    s.append("bedrooms")
if "toilets" in data.columns:
    s.append("toilets")
if "parking_space" in data.columns:
    s.append("parking_space")

# Check if any variables are left to plot
if not s:
    print("No valid variables to plot.")
else:
    # Remove missing values
    data_cleaned = data[s].dropna()

    if data_cleaned.shape[1] < 2:
        print("Insufficient data to create a pairplot with 2 or more variables.")
    else:
        # Create the pairplot
        plot = sns.pairplot(data_cleaned)

        # Check if "state" exists in the DataFrame
        if "state" in data.columns:
            # Create a jointplot with KDE using valid data and "state" as hue
            g = sns.jointplot(
                data=data_cleaned,
                x="bathrooms", y="bedrooms", hue='parking_space', kind="kde"
            )

st.pyplot(plot)
st.pyplot(g)

# if bathrooms:
#     s.append("bathrooms")

# if bedrooms:
#     s.append("bedrooms")

# if toilets:
#     s.append("toilets")

# if parking_space:
#     s.append('parking_space')


# plot=sns.pairplot(data[s])


# g = sns.jointplot(
#     data=data,
#     x="bathrooms", y="bedrooms", hue="state",
#     kind="kde",
#     )

# x=sns.catplot(
#     data=data, x="parking", y="price",
#     kind="violin", split=True, palette="pastel",
# )

y=sns.relplot(data=data, x="bathrooms", y="bedrooms",hue="price")


# st.pyplot(plot)

# st.pyplot(g)
# st.pyplot(x)
st.pyplot(y)