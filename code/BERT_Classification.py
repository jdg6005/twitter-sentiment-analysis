import torch
from transformers import Trainer, TrainingArguments, pipeline, DistilBertForSequenceClassification, DistilBertTokenizer

from preprocessing import preprocess_data
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def get_results(tt_split):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained("./results/checkpoint-14800/")
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        eval_strategy="epoch",
        logging_dir="./logs",
    )
    X_train_full, X_test_full, y_train, y_test = tt_split

    X_train_full = X_train_full.reset_index(drop=True)
    X_test_full = X_test_full.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    label_map = {label: idx for idx, label in enumerate(y_train.unique())}
    y_test = y_test.map(label_map)
    test_tokens = tokenizer(
        list(X_test_full),
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    test_dataset = TweetDataset(test_tokens, y_test)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate()
    print(results)

def run_BERT(tt_split):
    X_train_full, X_test_full, y_train, y_test = tt_split

    X_train_full = X_train_full.reset_index(drop=True)
    X_test_full = X_test_full.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    #changes labels to ints
    label_map = {label: idx for idx, label in enumerate(y_train.unique())}
    y_train = y_train.map(label_map)
    y_test = y_test.map(label_map)

    #set up model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(label_map)
    )

    #tokenize
    train_tokens = tokenizer(
        list(X_train_full),
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=128
    )

    test_tokens = tokenizer(
        list(X_test_full),
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    train_dataset = TweetDataset(train_tokens, y_train)
    test_dataset = TweetDataset(test_tokens, y_test)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        eval_strategy="epoch",
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()
    print(results)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist() 

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

df, tt = preprocess_data()
get_results(tt)


"""
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
i = 0 #number of sentiments found
for tweet, sentiment in zip(tweets, sentiments):
    i+= 1
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
    print(i)
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


    if(i == 1000):
        break
    

accuracy = accuracy / i
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
"""