import pandas as pd 
import numpy as np 
import pickle
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('news.csv')
df.columns = ['id','title','text','label']
# print(df.shape)
# print(df.head())

labels = df.label
X_train, X_test, y_train, y_test = train_test_split(df['text'],labels,test_size=1)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
tfidf_test  = tfidf_vectorizer.transform(X_test)

# Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

# saving vectorizer
with open('tfid.pickle','wb') as f:
    pickle.dump(tfidf_vectorizer,f)

# saving model
with open('model_fakenews.pickle','wb') as f:
    pickle.dump(pac,f)