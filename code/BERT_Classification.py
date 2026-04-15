import torch
from transformers import pipeline
from transformers import BertConfig, BertModel
import preprocessing
df, tfidf_matrix, feature_names = preprocessing.preprocess_data()
tweets = df["tweet"]
sentiments = df["sentiment"]
configuration = BertConfig()
model = BertModel(configuration)
configuration = model.config
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
#T is true, F is false, P is positive, NT is Neutral, NG is negative
TP = 0
FP = 0
TNT = 0
FNT = 0
TNG = 0
FNG = 0
P = 0
NT = 0
NG = 0 

accuracy = 0 #number of correct sentiments
count = 0 #number of sentiments found
for tweet, sentiment in zip(tweets, sentiments):
    count+= 1
    result = sentiment_analyzer(tweet)
    if sentiment == "Irrelevant":
        sentiment = "Neutral"
    if sentiment == "Positive":
        P+=1
    elif sentiment == "Neutral":
        NT+=1
    elif sentiment == "Negative":
        NG+=1
    prediction = result[0]["label"]
    if result[0]["score"] < 0.70:
        prediction = "Neutral"
    print(count)
    if(prediction.lower() == sentiment.lower()):
        accuracy += 1
        if(prediction.lower() == "positive"):
            TP+=1
        elif(prediction.lower() == "negative"):
            TNG+=1
        elif(prediction.lower() == "neutral"):
            TNT+=1
    else:
        if(prediction.lower() == "positive"):
            FP+=1
        elif(prediction.lower() == "negative"):
            FNG+=1
        elif(prediction.lower() == "neutral"):
            FNT+=1


    if(count == 1000):
        break
    

accuracy = accuracy / count
print("Accuracy: ", accuracy)
precisionP = TP/(TP + FP)
recallP = TP/P
f1P = 2*(precisionP*recallP)/(precisionP+recallP)
precisionNT = TNT/(TNT + FNT)
recallNT = TNT/NT
f1NT = 2*(precisionNT*recallNT)/(precisionNT+recallNT)
precisionNG = TNG/(TNG + FNG)
recallNG = TNG/NG
f1NG = 2*(precisionNG*recallNG)/(precisionNG+recallNG)
averagePrecision = (precisionNG + precisionNT + precisionP)/3
averageRecal = (recallNG + recallNT + recallP)/3
averageF1 = 2*(averagePrecision*averageRecal)/(averagePrecision+averageRecal)
print("Positive precision: ", precisionP)
print("Positive recall: ", recallP)
print("Positive F1: ", f1P)
print("Neutral precision: ", precisionNT)
print("Neutral recall: ", recallNT)
print("Neutral F1: ", f1NT)
print("Negative precision: ", precisionNG)
print("Negative recall: ", recallNG)
print("Negative F1: ", f1NG)
print("Average precision: ", averagePrecision)
print("Average recall: ", averageRecal)
print("Average F1: ", averageF1)

#accuracy, precision, recall, f1