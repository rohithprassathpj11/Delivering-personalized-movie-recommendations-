import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD

nltk.download('vader_lexicon')

# Load datasets
movies = pd.read_csv("movies.csv")  # Columns: movieId, title, genres
ratings = pd.read_csv("ratings.csv")  # Columns: userId, movieId, rating
sia = SentimentIntensityAnalyzer()

# TF-IDF for genres
tfidf = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Collaborative Filtering Model
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
algo = SVD()
algo.fit(data.build_full_trainset())

def sentiment_score(text):
    if text:
        return sia.polarity_scores(text)['compound']
    return 0

def hybrid_recommend(user_id, movie_title, review_text="", top_n=5):
    idx = indices.get(movie_title, None)
    if idx is None:
        return []

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+15]
    movie_indices = [i[0] for i in sim_scores]
    similar_movies = movies.iloc[movie_indices].copy()

    # Add collaborative predictions
    similar_movies['est_rating'] = similar_movies['movieId'].apply(lambda x: algo.predict(user_id, x).est)

    # Adjust with sentiment
    sentiment = sentiment_score(review_text)
    similar_movies['est_rating'] += sentiment

    recommendations = similar_movies.sort_values('est_rating', ascending=False).head(top_n)
    return recommendations[['title', 'genres', 'est_rating']].to_dict(orient='records')