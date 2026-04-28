# Twitter Sentiment Analysis
CSCI 335 Machine Learning Final Project

## Title
**Twitter Sentiment Analysis**

## Abstract
This project tests our Naive Bayes, Support Vector Machines, and Bert classification models on a twitter dataset.  The data is first
preprocessed via data cleaning (dropping n/a values) and splitting the data.  The goal is to test these various models and determine
how well they work via F1-score, accuracy, precision, recallm and confusion matrices.

## List of Developers
- Jaidan Giglio
- Logan Costa
- Harry Rinaudo

## How to Run
For naive bayes and bert classification, the programs can just be run via the built in run button in an IDE, or by running python code/filename.py.  For SVM, a jupyter notebook exists, where you can run each block sequentially.

### Prerequisites
- Python 3.8 or higher
- pip package manager


### Project Structure
```
twitter-sentiment-analysis/
├── code/
│   ├── BERT_Classification.py   # BERT model implementation
│   ├── Naive_Bayes.py           # Naive Bayes classifier
│   ├── preprocessing.py         # Data preprocessing module
│   └── SVM/
│       └── svm.ipynb            # SVM classifier notebook
├── data/
│   ├── twitter_training.csv     # Training data
│   └── twitter_validation.csv   # Validation data
└── README.md
```
