import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests


movies = pd.read_csv("https://raw.githubusercontent.com/shubhamjangid1995/shubhamdots/main/tmdb_movies.csv")

movies['tags']  = movies['overview'] + movies['genre']

movies = movies[['id','title', 'tags']]

cv =CountVectorizer(max_features=10000, stop_words='english')
#here we have given max_feature = 10000, because our dataset has 10000 rows of data.

vector = cv.fit_transform(movies['tags'].values.astype('U')).toarray()

similarity = cosine_similarity(vector)

st.set_page_config(page_title ="Movie Recommendation System",page_icon=":tada", layout="wide")




# movies = pickle.load(open("movies.pkl", "rb"))

# similarity = pickle.load(open("similarities.pkl", "rb"))

st.header("Movie Recommender System")

imagepath = "/home/shubham/projects/Recommendation_system/carousal.png"
st.image(imagepath)

#creating a dropdown list of movies

movie_list = movies['title']


select_value = st.selectbox("Please type or select a movie name from the dropdown list"
                            ,options=movie_list, index=0)

def fetch_posters(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=04d308196aea00396222e69f93ab5fe2".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/"+poster_path
    return full_path


#RECOMMEND FUNCTION

def recommend(movie):
    try:
        index = movies[movies['title']==movie].index[0]
        distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector:vector[1])
        recommend_movie=[]
        recommend_poster=[]
        for i in distance[1:11]:
            movie_id = movies.iloc[i[0]].id
            recommend_movie.append(movies.iloc[i[0]].title)
            recommend_poster.append(fetch_posters(movie_id))
        return recommend_movie, recommend_poster
    except:
        print("Please enter the correct name of the movie.")

if st.button("Show movie recommendations"):
    movie_name, movie_poster = recommend(select_value)

    
    with st.container():
        col1, col2, col3, col4, col5 =  st.columns(5)
        with col1:
            st.write(movie_name[0])
            st.image(movie_poster[0])
            
        with col2:
            st.write(movie_name[1])
            st.image(movie_poster[1])

        with col3:
            st.write(movie_name[2])
            st.image(movie_poster[2])

        with col4:
            st.write(movie_name[3])
            st.image(movie_poster[3])

        with col5:
            st.write(movie_name[4])
            st.image(movie_poster[4])

    st.markdown("##")

    with st.container():
        col6, col7, col8, col9, col10 = st.columns(5)

        with col6:
            st.write(movie_name[5])
            st.image(movie_poster[5])
            
        with col7:
            st.write(movie_name[6])
            st.image(movie_poster[6])

        with col8:
            st.write(movie_name[7])
            st.image(movie_poster[7])

        with col9:
            st.write(movie_name[8])
            st.image(movie_poster[8])

        with col10:
            st.write(movie_name[9])
            st.image(movie_poster[9])
