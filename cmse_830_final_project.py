
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import folium
import warnings


data = pd.read_csv("nigeria_houses_data.csv")
data_cords = pd.read_csv("dataset_with_coordinates.csv")
# mean_vals = data.groupby('state').mean()
# subdata = mean_vals.reset_index()


##################################################################################################################################################################################
st.sidebar.markdown(" # Currency Converter")
conversion_rate = st.sidebar.number_input("This currency converted defaults to Naira. To use this, please enter the Conversion Rate of your preferred currency below. (For example the conversion to US Dollars would be .0013.)", format="%.4f", value=1.0000, step=None)
currency_name = st.sidebar.text_input("To keep track of what currency you're observing, Enter Currency Name Below:", "Naira")
data['converted_price'] = data['price'] * conversion_rate

########################################################
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Introduction", "Housing Types", "Explorations", "Housing Cost Predictor", "Conclusions", "Developer Bio"])

with tab1:
    st.markdown(" # Nigeria Housing Predictor")
    #image
    image = Image.open("project_photo.jpg")
    st.image(image, caption="Lagos Island skyline from Victoria Island")
    st.markdown("This web app will allow you to visualize housing prices in Nigeria based on the state, town, and various other factors. Nigeria’s economic system is heavily negotiation-based. Whether you're in the market buying clothing or looking for a house, almost every price in Nigeria is negotiable. Currently, some severe issues with the country’s economy leave locals and foreigners vulnerable to unreasonable housing costs.  With this web app, those not accustomed to the typical way of negotiations can first view housing prices in specific areas to avoid getting talked into paying more for less.  Additionally, this app may be able to use the app to visualize which areas are financially feasible for their living.")
    st.markdown("For those who are potentially not from Nigeria or just looking to investigate different currencies, there is a currency converter on the left sidebar of the screen. Through certain portions of this app there will be an opportunity to display the data through a converted rate. To use the currency converter, you can input any conversion rate into the first box. For example, the conversion to US dollars is 0.0013. The purpose of the second box is to keep track of what currency rate you input in the first box, so you can input rates such as 'USD' or 'Euros' in that field for reference.")



with tab2:
    st.header("Welcome to the House Types Visualizer!")
    st.markdown("Are you curious about the various housing styles in Nigeria and want to understand what different houses look like based on local terminology? This webpage is designed to help you navigate the rich diversity of residential architecture in Nigeria and avoid any confusion when discussing house types. Explore the visual representations of various house types, from flats to bungalows and more, and gain a better understanding of the terminology used in Nigeria's housing landscape. Let's make housing terminology less confusing and more accessible for everyone. Start exploring now to broaden your knowledge of Nigerian housing styles!")


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
    st.markdown(" Welcome to The housing price exploration page. Here, you can investigate property prices based on various factors, such as location and amenities. To view the data in your preferred currency, simply navigate to the 'Currency Converter' tab on the left.")
    
    #Average Housing Price by State
    st.header("Average Housing Price by State")
    st.markdown("Explore average housing prices by state in Nigeria with this interactive bar chart, revealing insights into regional property costs. Hover over any of the bars to see the specific average price of each state.")
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
    if conversion_rate == 1.00:
        st.write("Mean Prices by State (Naira):")
        st.write(mean_prices_by_state_org)
    else:
        col1, col2,= st.columns([3,3])
        with col1: 
            st.write("Mean Prices by State (Naira):")
            st.write(mean_prices_by_state_org)
        with col2: 
            st.write("Mean Prices by State (Converted to {}):".format(currency_name))
            st.write(mean_prices_by_state)

    #Average Housing Price by Town
    st.header("Average Housing Price by Town")
    st.markdown("Explore average housing prices by town in Nigeria with this interactive bar chart, revealing insights into regional property costs. Hover over any of the bars to see the specific average price of each town.")
    mean_prices_by_town = data.groupby('town')['converted_price'].mean().sort_values(ascending=True)
    mean_prices_by_town_org = data.groupby('town')['price'].mean().sort_values(ascending=True)
    num_town = len(mean_prices_by_town)
    cmap = cm.get_cmap("viridis", num_town)
    chart2 = altair.Chart(mean_prices_by_town.reset_index()).mark_bar().encode(
        x='town:N',
        y='converted_price:Q',
        color=altair.Color('converted_price:Q', scale=altair.Scale(scheme='viridis'))
    ).properties(width=600)
    st.altair_chart(chart2, use_container_width=True)
    
    # st.bar_chart(mean_prices_by_town, use_container_width=True)
    if st.button("click to read town findings."):
        st.write("From this chart we see that Ikoyoi in Lagos,Nigeria has a highest prices by town. Ikoyi has an area called Banana Island, which is home of the wealthiest people in the entire country.")
    if conversion_rate == 1.00:
        st.write("Mean Prices by Town (Naira):")
        st.write(mean_prices_by_town_org)
    else: 
        col1, col2,= st.columns([3,3])
        with col1: 
            st.write("Mean Prices by Town (Naira):")
            st.write(mean_prices_by_town_org)
        with col2: 
            st.write("Mean Prices by Town (Converted to {}):".format(currency_name))
            st.write(mean_prices_by_town)

    #Housing Types Average Price
    st.header("Average Housing Price by House Type")
    st.markdown("Explore average housing prices by houst type in Nigeria with this interactive bar chart, revealing insights into costs by house type. Hover over any of the bars to see the specific average price. If you need further understanding about what each type of house looks like, nagivate to the Housing Types tab at the top of this page.")
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
    if conversion_rate == 1.00:
        st.write("Mean Prices by House Type (Naira):")
        st.write(mean_prices_by_title_org)
    else:
        col1, col2,= st.columns([3,3])
        with col1: 
            st.write("Mean Prices by House Type (Naira):")
            st.write(mean_prices_by_title_org)
        with col2: 
            st.write("Mean Prices by House Type (Converted to {}):".format(currency_name))
            st.write(mean_prices_by_title)
    
    # #Correlation Heatmap
    # st.header("Correlation Heatmap")
    # st.markdown("Explore correlations found between the numerical features in the data. All correlations are interpreted using the absolute value of the number.")
    # datacor = data.iloc[: , :-1]
    # fig, ax = plt.subplots()
    # sns.heatmap(datacor.corr(), annot= True, ax=ax)
    # st.pyplot(fig)
    # if st.button("Click to read the heatmap findings."):
    #     st.write("The heatmap tells us that bedrooms and bathrooms are most correlated with price.")
    
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
    st.markdown("This Housing Cost Estimator lets you explore the average price of properties based on your specific criteria. Whether you're looking for a home with a certain number of bedrooms, bathrooms, parking spaces, or toilets, this tool provides you with valuable insights. Find out the average price in different towns and states to make an informed decision when searching for your next home. Get started by adjusting the sliders and discover your approximate housing cost. Below the sliders, costs, and locations, you'll find a map that pinpoints the exact states and towns for that price. Get started below!")

    # data_cords['town_state'] = data_cords['town'] + ', ' + data_cords['state']

    num_bath = st.slider("Number of Bathrooms", min_value=1, max_value=9, value=3)
    num_bed = st.slider("Number of Bedrooms", min_value=1, max_value=9, value=3)
    num_park = st.slider("Number of Parking Spaces", min_value=1, max_value=9, value=3)
    num_toilet = st.slider("Number of Toilets", min_value=1, max_value=9, value=3)
   
    # model_data = pd.read_csv("modeling_data.csv")
    model_data_cord = pd.read_csv("modeling_data_cords.csv")
    numerical_features = ['bedrooms', 'bathrooms', 'toilets', 'parking_space', 'latitude', 'longitude']
    X = model_data_cord[numerical_features]
    y = model_data_cord['price']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize the linear regression model
    model = GradientBoostingRegressor()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    # predictions = model.predict(X_test)
    input_data = pd.DataFrame([[num_bed, num_bath, num_park, num_toilet, 0, 0]], columns=numerical_features)

    # Ensure 'latitude' and 'longitude' columns are present in input_data
    if 'latitude' not in input_data.columns or 'longitude' not in input_data.columns:
    # Add 'latitude' and 'longitude' columns from the original dataset
        input_data[['latitude', 'longitude']] = model_data_cord[['latitude', 'longitude']]

    input_data['latitude'] = model_data_cord['latitude'].mean()
    input_data['longitude'] = model_data_cord['longitude'].mean()


    input_data_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_data_scaled)[0]
    input_data['predicted_price'] = predicted_price

    st.write(f'Predicted Housing Price : {predicted_price:,.2f}')

    if conversion_rate != 1: 
        st.write(f'Predicted Price in {currency_name}: {predicted_price*conversion_rate:,.2f}')
#### ABOVE HERE IS GOOD 
# Make predictions on user input
    input_data = pd.DataFrame([[num_bed, num_bath, num_park, num_toilet, 0, 0]], columns=numerical_features)
    input_data[['latitude', 'longitude']] = model_data_cord[['latitude', 'longitude']]  # Add 'latitude' and 'longitude' columns from the original dataset

    # Ensure 'latitude' and 'longitude' columns are present in input_data
    if 'latitude' not in input_data.columns or 'longitude' not in input_data.columns:
        # Add 'latitude' and 'longitude' columns from the original dataset
        input_data[['latitude', 'longitude']] = model_data_cord[['latitude', 'longitude']]

    # Set latitude and longitude to the mean values for better map display
    input_data['latitude'] = model_data_cord['latitude'].mean()
    input_data['longitude'] = model_data_cord['longitude'].mean()

    input_data_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_data_scaled)[0]
    input_data['predicted_price'] = predicted_price

    # Display predicted housing price
    st.write(f'Predicted Housing Price: ${predicted_price:,.2f}')

    # Currency conversion (if needed)
    conversion_rate = 1  # replace with your actual conversion rate
    currency_name = "USD"  # replace with your desired currency name

    if conversion_rate != 1:
        st.write(f'Predicted Price in {currency_name}: ${predicted_price * conversion_rate:,.2f}')

    # Map visualization
    st.subheader('Price Distribution Map')

    # Display the map using st.map
    st.map(input_data)
    st.info("Blue marker indicates the predicted house location.") 

  
    # # Map visualization
    # st.subheader('Price Distribution Map')

    # # Display the map using st.map
    # st.map(input_data[['latitude', 'longitude']])
    # st.info("Blue marker indicates the predicted house location.")


    #map
     # Visualize on a map using Folium
    m = folium.Map(location=[model_data_cord['latitude'].mean(), model_data_cord['longitude'].mean()], zoom_start=12)

# Plot predicted location from user input
    for _, row in input_data.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f'Predicted Price: ${row["predicted_price"]:.2f}',
            icon=folium.Icon(color='red')
        ).add_to(m)

    m.save('predicted_prices_map.html')
    st.components.v1.html(open('predicted_prices_map.html').read(), height=600, width=800)  

    #     # Visualize on a map using Folium
    # m = folium.Map(location=[model_data_cord['latitude'].mean(), model_data_cord['longitude'].mean()], zoom_start=12)

    # # Plot predicted locations from user input
    # for _, row in input_data.iterrows():
    #     folium.Marker(
    #         location=[row['latitude'], row['longitude']],
    #         popup=f'Predicted Price: ${row["predicted_price"]:.2f}',
    #         icon=folium.Icon(color='red')
    #     ).add_to(m)

    # # Save or display the map
    # st.components.v1.html(open('predicted_prices_map.html').read(), height=600, width=800)
     # towns_with_avg_price = user_input['town_state'].unique()

        # st.write("Towns and States with the similar average prices:")
        # for town_state in towns_with_avg_price:
        #     st.write(f"- {town_state}")
        # user_input_rem = user_input[['latitude', 'longitude']].dropna()
        # st.map(user_input_rem.drop_duplicates(), use_container_width=True)
    import streamlit as st
    import pandas as pd
    import xgboost as xgb
    import folium

    df = pd.read_csv("modeling_data_cords.csv")

    # Separate features and target variable
    X = df[['bedrooms', 'bathrooms', 'parking_space', 'toilets']]
    y = df['price']

    # Train XGBoost model
    model = xgb.XGBRegressor()
    model.fit(X, y) 

    # Streamlit App
    st.title('House Price Prediction and Visualization')

    # User input for features
    bedrooms = st.slider('Bedrooms', min_value=1, max_value=9, value=2)
    bathrooms = st.slider('Bathrooms', min_value=1, max_value=9, value=2)
    parking_space = st.slider('Parking Space', min_value=0, max_value=9, value=2)
    toilets = st.slider('Toilets', min_value=1, max_value=9, value=2)

    # Make prediction
    prediction = model.predict([[bedrooms, bathrooms, parking_space, toilets]])
    st.subheader(f'Predicted Price: ${prediction[0]:,.2f}')

    # Map visualization
    st.subheader('Price Distribution Map')

    # Filter data for the selected features
    filtered_data = df[(df['bedrooms'] == bedrooms) & (df['bathrooms'] == bathrooms) &
                    (df['parking_space'] == parking_space) & (df['toilets'] == toilets)]
    # Display the map using st.map
    if not filtered_data.empty:
        st.map(filtered_data[['latitude', 'longitude']])
        st.info("Red markers indicate house locations.")
    else:
        st.warning("No data available for the selected features.")
    # # Create a folium map centered at the average latitude and longitude
    # if not filtered_data.empty:
    #     average_latitude = filtered_data['latitude'].mean()
    #     average_longitude = filtered_data['longitude'].mean()
        
    #     map = folium.Map(location=[average_latitude, average_longitude], zoom_start=12)

    #     # Mark areas on the map
    #     for index, row in filtered_data.iterrows():
    #         folium.Marker(location=[row['latitude'], row['longitude']],
    #                     popup=f"Price: ${row['price']:,.2f}",
    #                     icon=folium.Icon(color='blue')).add_to(map)

    #     # Display the map
    #     st.map(map, use_container_width=True)
    # else:
    #     st.warning("No data available for the selected features.")


#################################################################################################################################################################

with tab5: 
    imagefinal = Image.open("nigeria_pic.jpg")
    st.image(imagefinal)
    st.header("Conclusions")
    st.markdown("In conclusion, this web app is the result of a labor of love, crafted with the utmost dedication to provide you with a powerful tool to navigate Nigeria's housing market. As a developer, my primary goal has been to empower you, the user, with insights and knowledge that can be pivotal in your housing journey.")
    st.markdown("Through a collection of user-friendly visualizations, I've designed this app to be your partner in exploring the housing landscape. Whether you're in search of your dream home or considering selling your property, this app is here to equip you with critical information to make informed decisions.")
    st.markdown("As Nigeria faces economic challenges, this app comes at an opportune time. It places the power to understand housing prices firmly in your hands. You can now gauge what to expect before embarking on your housing search or during negotiations with real estate agents. It's all about ensuring you're well-prepared and well-informed in an ever-evolving real estate market.")
    st.markdown("The reach of this app isn't limited to just buyers and sellers; it extends to anyone who might be seeking accommodations. In today's dynamic world, where housing concerns touch virtually everyone at some point, the app's versatility becomes its defining feature. It's a resource that's ready to serve you, regardless of where you stand in the real estate landscape.")
    st.markdown("As you embark on your housing journey, remember that you're not alone. I'm here to support you with this app, and together, we can make navigating Nigeria's housing market a smoother, more informed, and ultimately, more successful experience.")

##################################################################################################################################################################

with tab6:
    st.markdown("add bio")