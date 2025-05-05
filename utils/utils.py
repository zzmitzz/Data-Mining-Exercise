import pandas as pd

"""
Load data and get the transaction list
"""
def getItemSetAndTransactionList(
        dataPath="ml-latest/cropped.csv", 
        minRating = 3.5):
    ratings = pd.read_csv(f"{dataPath}")
    ratings = ratings[ratings["rating"] >= minRating]
    buckets = ratings.groupby("userId")["movieId"].apply(list).tolist()
    itemSet = set((item,) for bucket in buckets for item in bucket)
    return itemSet, buckets