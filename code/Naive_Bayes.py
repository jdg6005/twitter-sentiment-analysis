import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
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

    model = MultinomialNB(alpha=0, force_alpha=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    # confusion matrix labels
    confuseL = unique_labels(y_test, y_pred)

    # confusion matrix
    confuseM = confusion_matrix(y_test, y_pred, labels=confuseL)

    # confusion matrix pandas dataframe
    confuseDF = pd.DataFrame(confuseM, index=[f"Actual {label}" for label in confuseL], columns=[f"Predicted {label}" for label in confuseL])

    confuseDisp = ConfusionMatrixDisplay(confuseM, display_labels=confuseL).plot(cmap=plt.cm.Blues)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Confusion Matrix:\n{confuseDF.to_string()}")

    plt.show()


if __name__ == "__main__":
    main()