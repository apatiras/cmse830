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
import altair

data = pd.read_csv("nigeria_houses_data.csv")
mean_vals = data.groupby('state').mean()
subdata = mean_vals.reset_index()


#introductory material 
st.markdown(" # Welcome! Explore Nigeria's Housing Market")
st.markdown("My project aims to create a web app that can visualize housing prices in Nigeria based on the state, town, and various other factors. Nigeria’s economic system is heavily negotiation-based. Whether in the market or looking for a house, almost every price in Nigeria is negotiable. Currently, some severe issues with the country’s economy leave locals and foreigners vulnerable to unreasonable rates.  With my web app, those not accustomed to the typical way of negotiations can first view housing prices in specific areas to avoid getting talked into paying more for less.  Additionally, locals may be able to use the app to visualize which areas are financially reasonable for their living. The dataset for this project contains 24,326 rows and 8 columns of housing data for Nigeria. ")
#image
image = Image.open("project_photo.jpg")
st.image(image, caption="Lagos Island skyline from Victoria Island")
#View data 
st.markdown(" Before beginning your exploration, please feel free to use the two follwing check boxes to view the data and basic statistics.")
col1, col2,= st.columns([3,3])
dfprint = col1.checkbox("Click to view the dataset")
if dfprint: 
    st.dataframe(data=data)
statprint = col2.checkbox("Click to view the basic data statistics")
if statprint:
    df_descr = data.describe(include="all")
    st.write(df_descr)

##################################################################################################################################################################################
st.sidebar.markdown(" # Currency Converter")
conversion_rate = st.sidebar.number_input("Dataframe defaults to prices in Naira. Enter Conversion Rate of preferred currency below: ", value=1.0000, step=0.001)
currency_name = st.sidebar.text_input("Enter Currency Name", "Naira")
data['converted_price'] = data['price'] * conversion_rate

st.sidebar.title("Page Navigation")
# st.sidebar.markdown("[Average Housing Price by State](#average_housing_price_by_state)")
# st.sidebar.markdown("[Average Housing Price by town](#average_housing_price_by_town)")
# st.sidebar.markdown("[Average Housing Price by House Type](#average_housing_price_by_title)")
# st.sidebar.markdown("[Nigeria Housing Dataset 3D Scatter Plot](#3D_Scatterplot)")
# st.sidebar.markdown("[Interactive Histogram of Characteristic Distribution](#interactive_histogram)")
# st.sidebar.markdown("[Housing Price Estimator](#estimator)")



st.markdown(" To first understand this dataset, explore the mean prices of houses by state")
################################################################################################################################################################
# Bar Chart By State
st.header("Average Housing Price by State")
if st.sidebar.button("Go to Housing Price by State"):
    st.markdown("[Jump to Average Housing Price by State](#average_housing_price_by_state)")


mean_prices_by_state_org = data.groupby('state')['price'].mean().sort_values(ascending=True)
mean_prices_by_state = data.groupby('state')['converted_price'].mean().sort_values(ascending=True)

st.bar_chart(mean_prices_by_state, use_container_width=True)

col1, col2,= st.columns([3,3])
with col1: 
    st.write("Mean Prices by State (Naira):")
    st.write(mean_prices_by_state_org)
with col2: 
    st.write("Mean Prices by State (Converted to {}):".format(currency_name))
    st.write(mean_prices_by_state)
###############################################################################################################################################
st.header("Average Housing Price by Town")
if st.sidebar.button("Go to Housing Price by Town"):
    st.markdown("[Jump to Average Housing Price by Town](#average_housing_price_by_town)")
mean_prices_by_town = data.groupby('town')['converted_price'].mean().sort_values(ascending=True)
mean_prices_by_town_org = data.groupby('town')['price'].mean().sort_values(ascending=True)
st.bar_chart(mean_prices_by_town, use_container_width=True)

col1, col2,= st.columns([3,3])
with col1: 
    st.write("Mean Prices by Town (Naira):")
    st.write(mean_prices_by_town_org)
with col2: 
    st.write("Mean Prices by Town (Converted to {}):".format(currency_name))
    st.write(mean_prices_by_town)
###############################################################################################################################################
st.header("Average Housing Price by House Type")
if st.sidebar.button("Go to Housing Price by House Type"):
    st.markdown("[Jump to Average Housing Price by House Type](#average_housing_price_by_title)")

mean_prices_by_title= data.groupby('title')['converted_price'].mean().sort_values(ascending=True)
mean_prices_by_title_org = data.groupby('title')['price'].mean().sort_values(ascending=True)

st.bar_chart(mean_prices_by_title, use_container_width=True)
col1, col2,= st.columns([3,3])
with col1: 
    st.write("Mean Prices by House Type (Naira):")
    st.write(mean_prices_by_title_org)
with col2: 
    st.write("Mean Prices by House Type (Converted to {}):".format(currency_name))
    st.write(mean_prices_by_title)

###############################################################################################################################################
#Correlation heatmap
st.header("Correlation Heatmap")
if st.sidebar.button("Go to Correlation Heatmap"):
    st.markdown("[Jump to Correlation Heatmap](#correlation_heatmap)")

fig, ax = plt.subplots()
sns.heatmap(data.corr(), annot= True, ax=ax)
st.write("Here is a heatmap that displays correlations of the data",fig)

##############################################################################################################################################
st.markdown("From the bar chart and Heatmap we see that Lagos and Abuja are the most expensive states in the housing market.Ikoyi in particular is the town with the highest housing prices.  The heatmap tells us that bedrooms and bathrooms are most correlated with price.")
###############################################################################################################################################
#interactive chart 
# 3D scatter plot using Plotly
st.header("Nigeria Housing Dataset 3D Scatter Plot")
if st.sidebar.button("Go to 3D Scatterplot"):
    st.markdown("[Jump to 3D Scatterplot](#3d_scatterplot)")

data_columns = list(data.columns.values)
options = st.multiselect("Pick three column paramaters for 3D Scatterplot", data_columns)
selected_state = st.multiselect("Select State to Display", data['state'].unique())

if len(options) >= 3:  
    filtered_data = data[data['state'].isin(selected_state)]
    fig = px.scatter_3d(filtered_data, x=options[0], y=options[1], z=options[2], color='state')
    st.write("This interactive 3D scatter plot visualizes the housing dataset with the selected axes.")
    st.plotly_chart(fig, use_container_width=True)
else: 
    st.write("Please select columns to create the 3D scatter plot.")

##################################################################################################################################################
# #Hiplot Visualization 
# st.header("Hiplot visualization")
# exp = hip.Experiment.from_dataframe(data)
# st_hiplot=st.empty()
# st.write(exp)
###################################################################################################################################################

# Interactive histogram
st.header("Interactive Histogram of Characteristic Distribution")
if st.sidebar.button("Go to Characteristic Distribution"):
    st.markdown("[Jump to Characteristic Histograms](#characteristic_histograms)")

data = pd.read_csv("nigeria_houses_data.csv")
characteristic = st.selectbox("Select a Housing Characteristic", ['bathrooms', 'bedrooms', 'toilets', 'parking_space'])

filtered_data = data[characteristic].dropna()  # Drop any NaN values

fig = ff.create_distplot([filtered_data],[characteristic], bin_size=0.1)
st.plotly_chart(fig, use_container_width=True)

st.write(f"Summary statistics for {characteristic}:")
st.write(filtered_data.describe())

######################################################################################################################################
st.header("Housing Price Estimator")
if st.sidebar.button("Go to Housing Price Estimator"):
    st.markdown("[Jump to Housing Price Estimator](#price_estimator)")

data = pd.read_csv("nigeria_houses_data.csv")
mean_vals = data.groupby('town').mean()
average_values = mean_vals.reset_index()


bedrooms = st.number_input("Bedrooms", min_value=1.0, max_value=9.0, step=0.5)
bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=9.0, step=0.5)
parking_spaces = st.number_input("Parking Spaces", min_value=0.0, max_value=9.0, step=0.5)
toilets = st.number_input("Toilets", min_value=1.0, max_value=9.0, step=0.5)

similar_houses = data[
    (data['bedrooms'] == bedrooms) &
    (data['bathrooms'] == bathrooms) &
    (data['parking_space'] == parking_spaces) &
    (data['toilets'] == toilets)
]


if not similar_houses.empty:
    average_price = similar_houses['price'].mean()
    st.write(f"Average Price for Similar Houses: ₦{average_price:.2f}")
else:
    st.write("No houses with similar characteristics found.")

# Find towns with similar prices
similar_towns = data[
    (data['price'] >= (average_price - 10000)) &
    (data['price'] <= (average_price + 10000))
]

if not similar_towns.empty:
    unique_towns = similar_towns['town'].unique()
    st.write("Towns with Similar Prices:")
    st.write(unique_towns)
else:
    st.write("No towns with similar prices found.")

