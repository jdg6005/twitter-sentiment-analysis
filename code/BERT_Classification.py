import pandas as pd
import numpy as np
import preprocessing



df, tfidf_matrix, feature_names = preprocessing.preprocess_data()

print(df)
print(tfidf_matrix)
print(feature_names)