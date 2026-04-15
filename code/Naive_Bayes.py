import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import preprocessing

def main():
    df, tt_split = preprocessing.preprocess_data()
    
    X_train_full, X_test_full, y_train, y_test = tt_split

    vector = TfidfVectorizer(ngram_range=(2, 2))
    X_train = vector.fit_transform(X_train_full)
    X_test = vector.transform(X_test_full)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    F1_score = f1_score(y_test, y_pred, average='weighted')

    # confusion matrix labels
    confuseL = unique_labels(y_test, y_pred)

    # confusion matrix
    confuseM = confusion_matrix(y_test, y_pred, labels=confuseL)

    # confusion matrix pandas dataframe
    confuseDF = pd.DataFrame(confuseM, index=[f"Actual {label}" for label in confuseL], columns=[f"Predicted {label}" for label in confuseL])

    confuseDisp = ConfusionMatrixDisplay(confuseM, display_labels=confuseL).plot(cmap=plt.cm.Blues)
    plt.show()
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {F1_score:.4f}")
    print(f"Confusion Matrix:\n{confuseDF.to_string()}")

if __name__ == "__main__":
    main()