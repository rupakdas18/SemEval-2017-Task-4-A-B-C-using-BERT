# Import Libraries
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score


# Load DataSets
train_data_df = pd.read_csv("data/twitter-2016train-A.txt", delimiter='\t', header=None, names=['id', 'label', 'tweet'])
test_data_df = pd.read_csv("data/twitter-2016test-A.txt", delimiter='\t', header=None, names=['label', 'tweet', 'id'])
#test_data_df = pd.read_csv("data/twitter-2016devtest-A.txt", delimiter='\t', header=None, names=['id', 'label', 'tweet'])

train_tweet = train_data_df.tweet.values
y_train = train_data_df.label.values

test_tweet = test_data_df.tweet.values
y_test = test_data_df.label.values


# Change labels in Numeric values
train_labels=[]
test_labels=[]
label_dict = {'negative':0, 'neutral':1, 'positive':2}

for label in y_train:
  train_labels.append(label_dict[label])

for label in y_test:
  test_labels.append(label_dict[label])
  
print("We have {} training samples".format(len(train_tweet)))
print("We have {} test samples".format(len(test_tweet)))  
  

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Use CountVectorization to convert text to a matrix of token counts
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_tweet)


# Use TfidfTransformer to convert into matrix of TF-IDF features
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# Train the Model
model = MultinomialNB().fit(X_train_tfidf, train_labels)


# Process the Test Data
X_test_counts = count_vect.transform(test_tweet)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)


from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Predict the Output
predicted = model.predict(X_test_tfidf)


# Find the Accuracy, precision, recall and F1 score
test_acc =accuracy_score(test_labels, predicted)
test_f1 = metrics.f1_score(test_labels, predicted, labels=[0, 1, 2], average='macro')
test_precision = precision_score(test_labels, predicted,labels=[0, 1, 2], average='macro')
test_recall = recall_score(test_labels, predicted,labels=[0, 1, 2], average='macro')

print(f'test_acc: {test_acc:.4f}')
print(f'f1 Score: {test_f1:.4f}')
print(f'precision: {test_precision:.4f}')
print(f'recall: {test_recall:.4f}')
