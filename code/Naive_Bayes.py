"""


"""


import math
import random
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score


def encode_class(mydata):
    classes = []
    for i in range(len(mydata)):
        if mydata[i][-1] not in classes:
            classes.append(mydata[i][-1])
    for i in range(len(classes)):
        for j in range(len(mydata)):
            if mydata[j][-1] == classes[i]:
                mydata[j][-1] = i
    return mydata


def splitting(mydata, ratio):
    train_num = int(len(mydata) * ratio)
    train = []
    test = list(mydata)

    while len(train) < train_num:
        index = random.randrange(len(test))
        train.append(test.pop(index))
    return train, test

def groupUnderClass(mydata):
    data_dict = {}
    for i in range(len(mydata)):
        if mydata[i][-1] not in data_dict:
            data_dict[mydata[i][-1]] = []
        data_dict[mydata[i][-1]].append(mydata[i][:-1])
    return data_dict

def MeanAndStdDev(numbers):
    avg = np.mean(numbers)
    stddev = np.std(numbers)
    return avg, stddev

def MeanAndStdDevForClass(mydata):
    info = {}
    data_dict = groupUnderClass(mydata)
    for classValue, instances in data_dict.items():
        info[classValue] = [MeanAndStdDev(attribute) for attribute in zip(*instances)]
    return info

def calculateGaussianProbability(x, mean, stdev):
    epsilon = 1e-10
    expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev + epsilon, 2))))
    return (1 / (math.sqrt(2 * math.pi) * (stdev + epsilon))) * expo

def calculateClassProbabilities(info, test):
    probabilities = {}
    for classValue, classSummaries in info.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, std_dev = classSummaries[i]
            x = test[i]
            probabilities[classValue] *= calculateGaussianProbability(x, mean, std_dev)
    return probabilities

def predict(info, test):
    probabilities = calculateClassProbabilities(info, test)
    bestLabel = max(probabilities, key=probabilities.get)
    return bestLabel

def getPredictions(info, test):
    predictions = [predict(info, instance[:-1]) for instance in test]
    return predictions


def accuracy_rate(test, predictions):
    correct = sum(1 for i in range(len(test)) if test[i][-1] == predictions[i])
    return (correct / float(len(test))) * 100.0


def main():
    df, tfidf_matrix, feature_names = preprocessing.preprocess_data()

    # Use sentiment as labels
    y = df["sentiment"].astype("category").cat.codes.to_numpy()

    # TF-IDF features
    X = tfidf_matrix.toarray()

    # Combine features + label
    mydata = np.column_stack([X, y]).tolist()

    ratio = 0.7
    train_data, test_data = splitting(mydata, ratio)

    print('Total number of examples:', len(mydata))
    print('Training examples:', len(train_data))
    print('Test examples:', len(test_data))

    info = MeanAndStdDevForClass(train_data)

    predictions = getPredictions(info, test_data)
    accuracy = accuracy_rate(test_data, predictions)
    print('Accuracy of the model:', accuracy)

    y_true = [row[-1] for row in test_data]
    y_pred = predictions

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.show()

    actual = [0, 1, 1, 0, 1, 0, 1, 1]
    predicted = [0, 1, 0, 0, 1, 0, 1, 0]

    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)

    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [precision, recall, f1]

    plt.figure(figsize=(6, 4))
    plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylim(0, 1)
    plt.title('Precision, Recall, and F1 Score')
    plt.ylabel('Score')
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

    plt.show()


if __name__ == "__main__":
    main()