#Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import re          #(regular expression)
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

# Read data
df= pd.read_csv("Language Detection.csv")
df.head()
df["Language"].value_counts()

# feature and label extraction
X = df["Text"]
y = df["Language"]

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# cleaning the data
text_list = []

# iterating through all the text
for text in X:         
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)      # removes all the symbols and numbers
    text = re.sub(r'[[]]', ' ', text)   
    text = text.lower()          # converts all the text to lower case
    text_list.append(text)       # appends the text to the text_list
    
# Encode the feature(text)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer() 
X = cv.fit_transform(text_list).toarray() # tokenize a collection of text documents and store it
                                            #in an array
# check the shape of the data   
X.shape

# split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80, random_state=32)

# Model Training
from sklearn.naive_bayes import MultinomialNB  
model = MultinomialNB()
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy is :",acc)


def predict(txt):
     lang = cv.transform([txt]).toarray() # convert text to bag of words model (Vector)
     language = model.predict(lang) # predict the language
     language = le.inverse_transform(language) # find the language corresponding with the predicted value
     print ("The language is in", language[0]) # printing the language

predict('I went home yesterday')  # Call the function
