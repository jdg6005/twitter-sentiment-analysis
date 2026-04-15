import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split




def preprocess_data():
    df = pd.read_csv("./data/twitter_training.csv", header=None) # path to data
    df.columns = ["id", "entity", "sentiment", "tweet"] # naming columns for easier access
    df = df.dropna()
    
    X = df["tweet"]
    y = df['sentiment']

    tt_split = train_test_split(X, y, test_size=0.2, random_state=35, stratify=y)

    return df, tt_split



