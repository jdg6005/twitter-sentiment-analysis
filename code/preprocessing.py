import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("./data/twitter_training.csv", header=None) # path to data
df.columns = ["id", "entity", "sentiment", "tweet"] # naming columns for easier access
df = df.dropna()

vectorizer = TfidfVectorizer(ngram_range=(2, 2))  
tfidf_matrix = vectorizer.fit_transform(df['tweet'])

feature_names = vectorizer.get_feature_names_out()
print("Feature Names:", feature_names)