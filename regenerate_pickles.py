import pandas as pd
import numpy as np
import ast
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Process release_date and extract year
movies['release_date'] = pd.to_datetime(movies['release_date'])
movies['year'] = movies['release_date'].dt.year

# Merge movies with credits on title
movies = movies.merge(credits, on='title')
movies.rename(columns={'movie_id_x': 'movie_id'}, inplace=True)

# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'year']]
movies['year'] = movies['year'].astype(str).str[:4]

# Drop null values
movies.dropna(inplace=True)

# Function to extract names from JSON-like strings
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# Extract genres and keywords
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Extract cast (first 4 actors)
def convert_cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 4:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

# Extract director and music composer from crew
def convert_crew(obj):
    L_dir = []
    L_music = []
    for i in ast.literal_eval(obj):
        if i['job'] and i['job'] == 'Director':
            L_dir.append(i['name'])
        if i['job'] and i['job'] == 'Original Music Composer':
            L_music.append(i['name'])
    return L_dir, L_music

movies['crew'] = movies['crew'].apply(convert_crew)

# Split overview into list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces from genres, keywords, and cast
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])

# Create director and music_composer columns
movies['director'] = movies['crew'].apply(lambda x: x[0])
movies['music_composer'] = movies['crew'].apply(lambda x: x[1])

# Remove spaces from director and music_composer names
movies['director'] = movies['director'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['music_composer'] = movies['music_composer'].apply(lambda x: [i.replace(" ", "") for i in x])

# Create tags column by combining all features
movies['tags'] = (movies['overview'] + movies['genres'] + movies['keywords'] + 
                  movies['cast'] + movies['director'] + movies['music_composer'])

# Create moviesdf with required columns
moviesdf = movies[['movie_id', 'title', 'tags']]
moviesdf['tags'] = moviesdf['tags'].apply(lambda x: " ".join(x))

# Convert tags to lowercase
moviesdf['tags'] = moviesdf['tags'].apply(lambda x: x.lower())

# Apply stemming to tags
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

moviesdf['tags'] = moviesdf['tags'].apply(stem)

# Create Count Vectorizers and fit vectors
cv5 = CountVectorizer(max_features=5000, stop_words='english')
vectors5 = cv5.fit_transform(moviesdf['tags']).toarray()

cv2 = CountVectorizer(max_features=2000, stop_words='english')
vectors2 = cv2.fit_transform(moviesdf['tags']).toarray()

# Calculate similarity matrices using cosine similarity
similarity5 = cosine_similarity(vectors5)
similarity2 = cosine_similarity(vectors2)

# Save to pickle files - matching the notebook format
pickle.dump(moviesdf, open('movies.pkl', 'wb'))
pickle.dump(similarity5, open('similarity.pkl', 'wb'))

print("✓ Pickle files regenerated successfully!")
print(f"  - movies.pkl: {len(moviesdf)} movies")
print(f"  - similarity.pkl: {similarity5.shape}")
