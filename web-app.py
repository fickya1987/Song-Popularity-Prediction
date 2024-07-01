
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore

# Title and description
st.write("""
# Song Popularity Prediction App
## This app can predict a song's popularity from 1960 - 2024!
""")


st.sidebar.header('User Input Parameters')

# Get user inputs
def user_input_features():
    acousticness = st.sidebar.slider('acousticness', 0.0, 1.0, 0.23)
    danceability = st.sidebar.slider('danceability', 0.0, 1.0, 0.7)
    energy = st.sidebar.slider('energy', 0.0, 1.0, 0.5)
    explicit = st.sidebar.slider('explicit', 0, 1, 1)
    instrumentalness = st.sidebar.slider('instrumentalness', 0.0, 1.0, 0.0)
    key = st.sidebar.slider('key', 0, 11, 3)
    liveness = st.sidebar.slider('liveness', 0.0, 1.0, 0.5)
    loudness = st.sidebar.slider('loudness', -60.0, 0.0, -4.0)
    mode = st.sidebar.slider('mode', 0, 1, 0)
    speechiness = st.sidebar.slider('speechiness', 0.0, 1.0, 0.03)
    tempo = st.sidebar.slider('tempo', 0.0, 250.0, 92.0)
    valence = st.sidebar.slider('valence', 0.0, 1.0, 0.7)
    album_release_year = st.sidebar.slider('year', 1960, 2024, 2015)
    data = {
        'Danceability': danceability,
        'Energy': energy,
        'Key': key,
        'Loudness': loudness,
        'Mode': mode,
        'Speechiness': speechiness,
        'Acousticness': acousticness,
        'Instrumentalness': instrumentalness,
        'Liveness': liveness,
        'Valence': valence,
        'Tempo': tempo,
        'Time Signature': 4,  # Assuming a default time signature
        'Explicit': explicit,
        'Album Release Year': album_release_year
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Show user inputs
st.subheader('User Input parameters')
st.write(df)

# Create Plotly plot
columns = ['Danceability', 'Energy', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness']
df_song_char = df.filter(items=columns)
y = df_song_char.values.tolist()[0]

fig = go.Figure(data=go.Bar(x=columns, y=y), layout_title_text='Audio Features from User Input')
st.plotly_chart(fig, use_container_width=True)

# Load the trained model and scaler
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Normalize the features
features_scaled = scaler.transform(df)

# Predict song popularity
prediction = model.predict(features_scaled)

st.subheader('Predicted Song Popularity')
result = 'Popular' if prediction[0] == 1 else 'Not Popular'
st.write(f'The song is predicted to be: {result}')
