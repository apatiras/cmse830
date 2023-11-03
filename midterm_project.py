import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import hiplot as hip
import json
import numpy as np
import plotly.figure_factory as ff
import altair

data = pd.read_csv("nigeria_houses_data.csv")
data_cords = pd.read_csv("dataset_with_coordinates.csv")
mean_vals = data.groupby('state').mean()
subdata = mean_vals.reset_index()


##################################################################################################################################################################################
st.sidebar.markdown(" # Currency Converter")
conversion_rate = st.sidebar.number_input("Dataframe defaults to prices in Naira. Enter Conversion Rate of preferred currency below: ", value=1.0000, step=0.001)
currency_name = st.sidebar.text_input("Enter Currency Name", "Naira")
data['converted_price'] = data['price'] * conversion_rate

# st.sidebar.title("Page Navigation")

# 
########################################################
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introduction", "Housing Types", "Explorations", "Housing Cost Estimator", "Conclusions"])

with tab1:
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

with tab2:
    st.header("Welcome to our House Types Visualizer!")
    st.markdown("Are you curious about the various housing styles in Nigeria and want to understand what different houses look like based on local terminology? Our webpage is designed to help you navigate the rich diversity of residential architecture in Nigeria and avoid any confusion when discussing house types. Explore the visual representations of various house types, from flats to bungalows and more, and gain a better understanding of the terminology used in Nigeria's housing landscape. Let's make housing terminology less confusing and more accessible for everyone. Start exploring now to broaden your knowledge of Nigerian housing styles!")


    house_images = {
        'Block of Flats': {
            "images": ['flats_image.png', 'flats_image2.png', 'flats_image3.png'],
            "description": "A block of flats is a multi-story residential building consisting of multiple individual apartments or living units with a single structure."
        },
        'Detached Bungalow': {
            "images": ['detachedbung_img.png', 'detachedbung_img2.png', 'detachedbung_img3.png'],
            "description": "A detached bungalow is a single-story building with no shared walls, offering independent living spaces for a single household."
        },
        'Semi-detached Duplex': {
            "images": ['semidetached_img1.png', 'semidetached_img2.png', 'semidetached_img3.png'],
            "description": "A semi-detached duplex is a two-story residential building that shares one common wall with another unit, providing separate living spaces for two houesholds. "
        },
        'Terraced Duplex' : {
            "images" :['terracedup_img1.png', 'terracedup_img2.png', 'terracedup_img3.png'],
            "description": "A terraced duplex is a multi-story residential building with a row-house design featuring a balcony or porch. They typically share side walls with neighboring units, offering separate living spaces for multiple households. "
        },
        'Detached Duplex' : {
            "images": ['detacheddup_img1.png', 'detacheddup_img2.png', 'detacheddup_img3.png'],
            "description": "A detached duplex is typically two-story residential building with no shared walls, providing separate and independent living spaces for households."
        },
        'Semi-Detached Bungalow': {
            "images" : ['semidetachedbung_img1.png', 'semidetachedbung_img2.png', 'semidetachedbung_img3.png'],
            "description": "A semi-detached bungalow is a single-story residential building with a design that shares one common wall with another unit, typically offering independent living spaces for two households."
        },
        'Terraced Bungalow': {
            "images" : ['terracedbung_img1.png', 'terracedbung_img2.png', 'terracedbung_img3.png'],
            "description": "A terraced bungalow is a single-story residential building with a compact design, sharing one or both side walls with neighboring units, often featuring front and back gardens."
        }
    }
    
    selected_house = st.selectbox("Select a House Type", list(house_images.keys()))

    if selected_house in house_images:
        st.write(f"**{selected_house}**")
        st.write(house_images[selected_house]["description"])
        st.write(f"The following images are examples of {selected_house}:")
        st.image(house_images[selected_house]["images"], use_column_width=True)
    else:
        st.write("Please selected a house type from the list.")


with tab3: 
    st.markdown("# Explore the housing dataset through the following visualizations ")
    st.markdown(" Welcome to our housing price exploration page. Here, you can investigate property prices based on various factors, such as location and amenities. To view the data in your preferred currency, simply navigate to the 'Currency Converter' tab on the left.")
    
    #Average Housing Price by State
    st.header("Average Housing Price by State")
    st.markdown("Explore average housing prices by state in Nigeria with our interactive bar chart, revealing insights into regional property costs. Hover over any of the bars to see the specific average price of each state.")
    mean_prices_by_state_org = data.groupby('state')['price'].mean().sort_values(ascending=True)
    mean_prices_by_state = data.groupby('state')['converted_price'].mean().sort_values(ascending=True)
    num_states = len(mean_prices_by_state)
    cmap = cm.get_cmap("viridis", num_states)
    chart = altair.Chart(mean_prices_by_state.reset_index()).mark_bar().encode(
        x='state:N',
        y='converted_price:Q',
        color=altair.Color('converted_price:Q', scale=altair.Scale(scheme='viridis'))
    ).properties(width=600)
    st.altair_chart(chart, use_container_width=True)
    if st.button("Click to read state findings."):
        st.write("From this chart we see that Lagos and Abuja are the most expensive states in Nigera. Lagos being the most populus state in the country and Abuja being the capital of the country. Inversely, Katsina and Plateau are the states with the cheapest housing prices. ")
    col1, col2,= st.columns([3,3])
    with col1: 
        st.write("Mean Prices by State (Naira):")
        st.write(mean_prices_by_state_org)
    with col2: 
        st.write("Mean Prices by State (Converted to {}):".format(currency_name))
        st.write(mean_prices_by_state)

    #Average Housing Price by Town
    st.header("Average Housing Price by Town")
    st.markdown("Explore average housing prices by town in Nigeria with our interactive bar chart, revealing insights into regional property costs. Hover over any of the bars to see the specific average price of each town.")
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

    #Housing Types Average Price
    st.header("Average Housing Price by House Type")
    st.markdown("Explore average housing prices by houst type in Nigeria with our interactive bar chart, revealing insights into costs by house type. Hover over any of the bars to see the specific average price. If you need further understanding about what each type of house looks like, nagivate to the Housing Types tab at the top of this page.")
    mean_prices_by_title= data.groupby('title')['converted_price'].mean().sort_values(ascending=True)
    mean_prices_by_title_org = data.groupby('title')['price'].mean().sort_values(ascending=True)
    num_houses = len(mean_prices_by_title)
    cmap = cm.get_cmap("viridis", num_houses)
    chart = altair.Chart(mean_prices_by_title.reset_index()).mark_bar().encode(
        x='title:N',
        y='converted_price:Q',
        color=altair.Color('converted_price:Q', scale=altair.Scale(scheme='viridis'))
    ).properties(width=600)
    st.altair_chart(chart, use_container_width=True)
    if st.button("Click to read house findings."):
        st.write("From this chart we find that a detached duplex and semi-detached duplex have the highest housing costs. Duplexes in general have a trend of costing more than bungalows. So, if you're looking for housing on the cheaper side, it may be wise to start off with explorations for bungalows.")
    col1, col2,= st.columns([3,3])
    with col1: 
        st.write("Mean Prices by House Type (Naira):")
        st.write(mean_prices_by_title_org)
    with col2: 
        st.write("Mean Prices by House Type (Converted to {}):".format(currency_name))
        st.write(mean_prices_by_title)
    
    #Correlation Heatmap
    st.header("Correlation Heatmap")
    st.markdown("Explore correlations found between the numerical features in the data. All correlations are interpreted using the absolute value of the number.")
    datacor = data.iloc[: , :-1]
    fig, ax = plt.subplots()
    sns.heatmap(datacor.corr(), annot= True, ax=ax)
    st.pyplot(fig)
    if st.button("Click to read the heatmap findings."):
        st.write("The heatmap tells us that bedrooms and bathrooms are most correlated with price.")
    
    # 3D scatter plot using Plotly
    st.header("Nigeria Housing Dataset 3D Scatter Plot")
    st.markdown("Take your data exploration to the next dimension with this interactive 3D scatterplot. Dive into the characteristics of the dataset from a new perspective, gaining deeper insights and uncovering hidden patterns.")
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

    # Interactive histogram
    st.header("Interactive Histogram of Characteristic Distribution")
    st.markdown("Explore the distribution of housing characteristics with the interactive histogram below. This tool allows you to visualize the distribution of key housing features, helping you gain insights into property traits and their prevalence within your dataset.")
    data = pd.read_csv("nigeria_houses_data.csv")
    characteristic = st.selectbox("Select a Housing Characteristic", ['bathrooms', 'bedrooms', 'toilets', 'parking_space'])
    filtered_data = data[characteristic].dropna()  # Drop any NaN values
    fig = ff.create_distplot([filtered_data],[characteristic], bin_size=0.1)
    fig.update_xaxes(title_text=characteristic)
    st.plotly_chart(fig, use_container_width=True)
    if st.button("Click to view the summary statistics for your feature."):
        st.write(f"Summary statistics for {characteristic}:")
        st.write(filtered_data.describe())

with tab4: 
    st.header("Housing Price Estimator")
    st.markdown("This Housing Cost Estimator lets you explore the average price of properties based on your specific criteria. Whether you're looking for a home with a certain number of bedrooms, bathrooms, parking spaces, or toilets, this tool provides you with valuable insights. Find out the average price in different towns and states to make an informed decision when searching for your next home. Get started by adjusting the sliders and discover your ideal housing cost. Below the sliders, costs, and locations, you'll find a map to pinpoint the exact states and towns. Get started below!")

    data_cords['town_state'] = data_cords['town'] + ', ' + data_cords['state']

    num_bath = st.slider("Number of Bathrooms", min_value=1, max_value=9, value=3)
    num_bed = st.slider("Number of Bedrooms", min_value=1, max_value=9, value=3)
    num_park = st.slider("Number of Parking Spaces", min_value=1, max_value=9, value=3)
    num_toilet = st.slider("Number of Toilets", min_value=1, max_value=9, value=3)


    user_input = data_cords[
        (data_cords['bathrooms'] == num_bath) &
        (data_cords['bedrooms'] == num_bed) &
        (data_cords['parking_space'] == num_park) &
        (data_cords['toilets'] == num_toilet) 
    ]
    if user_input.empty:
        st.warning("No matching data found for the selected criteria.")
    else:
        average_price = user_input['price'].mean()
        st.write(f"Average Cost: ₦{average_price:,.2f}")

        towns_with_avg_price = user_input['town_state'].unique()

        st.write("Towns and States with the same average price:")
        for town_state in towns_with_avg_price:
            st.write(f"- {town_state}")
        
        st.map(user_input[['latitude', 'longitude']].drop_duplicates(), use_container_width=True)

with tab5: 
    imagefinal = Image.open("nigeria_pic.jpg")
    st.image(imagefinal)
    st.header("Conclusions")
    st.markdown("In conclusion, my web app is the result of a labor of love, crafted with the utmost dedication to provide you with a powerful tool to navigate Nigeria's housing market. As a developer, my primary goal has been to empower you, the user, with insights and knowledge that can be pivotal in your housing journey.")
    st.markdown("Through a collection of user-friendly visualizations, I've designed this app to be your partner in exploring the housing landscape. Whether you're in search of your dream home or considering selling your property, this app is here to equip you with critical information to make informed decisions.")
    st.markdown("As Nigeria faces economic challenges, this app comes at an opportune time. It places the power to understand housing prices firmly in your hands. You can now gauge what to expect before embarking on your housing search or during negotiations with real estate agents. It's all about ensuring you're well-prepared and well-informed in an ever-evolving real estate market.")
    st.markdown("The reach of this app isn't limited to just buyers and sellers; it extends to anyone who might be seeking accommodations. In today's dynamic world, where housing concerns touch virtually everyone at some point, the app's versatility becomes its defining feature. It's a resource that's ready to serve you, regardless of where you stand in the real estate landscape.")
    st.markdown("s you embark on your housing journey, remember that you're not alone. I'm here to support you with this app, and together, we can make navigating Nigeria's housing market a smoother, more informed, and ultimately, more successful experience.")