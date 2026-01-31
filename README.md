# Movie Recommendation System 🎬

This project is a content-based movie recommendation system built using the TMDB dataset.
It recommends similar movies based on movie metadata.

## Tech Stack
- Python
- Pandas
- Scikit-learn

## Dataset
- TMDB 5000 Movies Dataset

## Features
- Content-based filtering
- Cosine similarity
- Case-insensitive movie search
- Uses real-world movie data

## How It Works
Movie metadata (overview, genres, keywords, cast, crew) is combined and vectorized.
Cosine similarity is used to find and recommend similar movies.

## How to Run the Project
```bash
pip install pandas scikit-learn
python movie_recommender.py
