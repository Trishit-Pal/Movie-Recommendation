import os
import pickle
import pandas as pd
import streamlit as st
import requests



def fetch_poster(movie_id):
    url= "https://api.themoviedb.org/3/movie/{}?api_key=e749164ed1aa86a57548ab5e975a1251&language=en-US".format(movie_id)
    response= requests.get(url)
    data= response.json()
    full_path= "https://image.tmdb.org/t/p/w500/"+data['poster_path']
    return full_path

def recommend(movie):
    movie_index=movies[movies['title']==movie].index[0]
    distances=similarity[movie_index]
    distances=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])
    movies_list=distances[1:11]
    recommended_movies=[]
    recommended_movie_posters=[]
    for i in movies_list:
        movie_id= movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        #fetch image from API
        recommended_movie_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_movie_posters

st.title('Movie Recommender')
movies_dict= pickle.load(open('movies.pkl','rb'))
movies= movies_dict['movies_small']
similarity= pickle.load(open('similarity.pkl','rb'))
selected_movie_name= st.selectbox(
    'Type or select a movie you like',
    movies['title'].values
)

if st.button('Show Recommendations'):
    names,posters= recommend(selected_movie_name)
    col1, col2, col3, col4, col5, col6, col7,col8, col9, col10= st.columns(10)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])
    col6, col7, col8, col9, col10= st.columns(5)
    with col6:
        st.text(names[5])
        st.image(posters[5])
    with col7:
        st.text(names[6])
        st.image(posters[6])
    with col8:
        st.text(names[7])
        st.image(posters[7])
    with col9:
        st.text(names[8])
        st.image(posters[8])
    with col10:
        st.text(names[9])
        st.image(posters[9])
