import torch
from transformers import pipeline, BertTokenizer
import preprocessing
df, tfidf_matrix, feature_names = preprocessing.preprocess_data()
tweets = df["tweet"]
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

print("start")
tokenized_tweets = [tokenizer.tokenize(tweet) for tweet in tweets]


print(tweets)
print("done")