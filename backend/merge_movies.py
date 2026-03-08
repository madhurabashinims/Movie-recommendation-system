import pandas as pd

old_movies = pd.read_csv(r"D:\MINI PROJECT\DATASET\movies_final.csv")
new_movies = pd.read_csv(r"D:\MINI PROJECT\DATASET\new_tmdb_movies.csv")

combined = pd.concat([old_movies, new_movies], ignore_index=True)

combined.drop_duplicates(subset=["title"], inplace=True)

combined.to_csv(r"D:\MINI PROJECT\DATASET\movies_extended.csv", index=False)

print("Total movies after merge:", len(combined))