import pandas as pd

def split_ratings(ratings):

    ratings = ratings.sort_values("timestamp")

    split_index = int(len(ratings) * 0.8)

    train_ratings = ratings.iloc[:split_index]
    test_ratings = ratings.iloc[split_index:]

    print("Train size:", len(train_ratings))
    print("Test size:", len(test_ratings))

    return train_ratings, test_ratings