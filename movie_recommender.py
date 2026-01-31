import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on="title")

# Select useful columns
movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

# Fill missing values
movies.fillna("", inplace=True)

# Combine features into one column
movies["tags"] = (
    movies["overview"]
    + " "
    + movies["genres"]
    + " "
    + movies["keywords"]
    + " "
    + movies["cast"]
    + " "
    + movies["crew"]
)

# Convert to lowercase
movies["tags"] = movies["tags"].str.lower()

# Vectorization
vectorizer = CountVectorizer(max_features=5000, stop_words="english")
vectors = vectorizer.fit_transform(movies["tags"]).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)

# Recommendation function (FINAL, CLEAN VERSION)
def recommend_movie(movie_name):
    movie_name = movie_name.lower()

    # Create lowercase title column once
    movies["title_lower"] = movies["title"].str.lower()

    if movie_name not in movies["title_lower"].values:
        return "Movie not found in database."

    index = movies[movies["title_lower"] == movie_name].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )

    recommendations = []
    for i in movie_list[1:6]:
        recommendations.append(movies.iloc[i[0]].title)

    return recommendations

# User input loop
while True:
    movie = input("\nEnter a movie name (or type exit): ")
    if movie.lower() == "exit":
        break

    result = recommend_movie(movie)
    print("Recommended movies:", result)
