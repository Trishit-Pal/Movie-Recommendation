import pandas as pd
import numpy as np
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load data
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Process movies data
movies['release_date'] = pd.to_datetime(movies['release_date'])
movies['year'] = movies['release_date'].dt.year

# Merge datasets
movies = movies.merge(credits, on='title')
movies.rename(columns={'movie_id_x': 'movie_id'}, inplace=True)

# Select columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'year']]

# Clean data
movies.dropna(inplace=True)
movies['year'] = movies['year'].fillna(0).astype(str).str[:4]

# Convert genres, keywords, cast, crew.
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

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

def convert_crew(obj):
    L_dir = []
    L_music = []
    for i in ast.literal_eval(obj):
        if i['job'] and i['job'] == 'Director':
            L_dir.append(i['name'])
        if i['job'] and i['job'] == 'Original Music Composer':
            L_music.append(i['name'])
    return L_dir, L_music

# Apply conversions
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(convert_crew)

# Split overview into list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])

# Extract director and music composer
movies['director'] = movies['crew'].apply(lambda x: x[0] if len(x) > 0 and x[0]else ['UnknownDirector']  )
movies['music_composer'] = movies['crew'].apply(lambda x: x[1] if len(x) > 1 and x[1] else ['UnknownComposer']  )

movies['director'] = movies['director'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['music_composer'] = movies['music_composer'].apply(lambda x: [i.replace(" ", "") for i in x])

# Create tags
movies['tags'] = (movies['overview'] + movies['genres'] + movies['keywords'] + 
                  movies['cast'] + movies['director'] + movies['music_composer'])

# Create moviesdf for similarity computation
moviesdf = movies[['movie_id', 'title', 'tags']].copy()
moviesdf['tags'] = moviesdf['tags'].apply(lambda x: " ".join(x))
moviesdf['tags'] = moviesdf['tags'].apply(lambda x: x.lower())

# Stemming - normalize words to their root form (love, loving, loved -> love)
print("Applying stemming..")
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

moviesdf['tags'] = moviesdf['tags'].apply(stem)
print("Stemming complete!")

# Create vectors
print("Creating feature vectors...")
cv5 = CountVectorizer(max_features=5000, stop_words='english')
vectors5 = cv5.fit_transform(moviesdf['tags']).toarray()
print(f"Feature extraction complete: {vectors5.shape}")

# Create similarity matrix
print("Computing similarity matrix... this may take a minute")
similarity5 = cosine_similarity(vectors5)
print(f"Similarity matrix complete: {similarity5.shape}")

movies_info = movies[['movie_id', 'title', 'year']].reset_index(drop=True)
moviesdf = moviesdf.reset_index(drop=True)  # Ensure same indexing

# Save pickle files
print("Saving pickle files...")

pickle.dump(moviesdf, open('movies.pkl', 'wb'))
pickle.dump(similarity5, open('similarity.pkl', 'wb'))
pickle.dump(movies_info, open('movies_info.pkl', 'wb'))

print("Pickle files regenerated successfully!")

