import streamlit as st
import time
import seaborn as sns
import pandas as pd


col1, col2, col3 = st.columns([1,2,1])

col1.markdown("# Welcome to my app!")
col1.markdown("This is the Best App You've Ever Seen")

cd = pd.read_csv("US_birthrates_by_race.csv")
cd_columns = list(cd.columns.values)
option1 = st.selectbox('Choose a x-value',(cd_columns))
option2 = st.selectbox('Choose a y-value',(cd_columns))
st.write('You selected:', option1, "and", option2)
# plot = sns.pairplot(cd[option1,option2])
st.pyplot(sns.relplot(data=cd, x= option1,y=option2,hue="Race"))



if "photo" not in st.session_state:
    st.session_state['photo']='not done'



def change_photo_state():
    st.session_state['photo']='done'

upload_photo = col2.file_uploader("Upload a Photo", on_change=change_photo_state)
camera_photo = col2.camera_input("Take a photo", on_change=change_photo_state)

if st.session_state['photo'] == 'done':
    progress_bar = col2.progress(0)

    for perc_completed in range(100):
        time.sleep(0.05)
        progress_bar.progress(perc_completed+1)
        

    col2.success("Photo uploaded successfully")

    col3.metric(label="temperature", value="60 C", delta="3 C")

    with st.expander("Click to read more"):
        st.write("Hello, here are more details on this topic.")

        if upload_photo is None:
            st.image(camera_photo)
        else:
            st.image(upload_photo)