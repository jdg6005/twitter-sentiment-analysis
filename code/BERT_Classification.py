import torch
from transformers import pipeline
import preprocessing
df, tfidf_matrix, feature_names = preprocessing.preprocess_data()
tweets = df["tweet"]
sentiments = df["sentiment"]

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
accuracy = 0 #number of correct sentiments
count = 0 #number of sentiments found
for tweet, sentiment in zip(tweets, sentiments):
    count+= 1
    result = sentiment_analyzer(tweet)
    print(count)
    if(result[0]["label"] == sentiment):
        accuracy += 1
    
#finds true accuracy
accuracy = accuracy / count
print(accuracy)


