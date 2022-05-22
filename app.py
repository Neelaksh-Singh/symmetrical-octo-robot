import numpy as np
import pandas as pd
from flask import Flask, redirect, request, render_template
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from langdetect import detect,DetectorFactory
from iso639 import languages
DetectorFactory.seed=0

cv = CountVectorizer()
le = LabelEncoder()

app = Flask(__name__)
model = joblib.load('Language-Detection-Model')

def predict(txt):
    
    t_o_b = cv.transform([txt]).toarray()# convert text to bag of words model (Vector)
    language = model.predict(t_o_b) # predict the language
    corr_language = le.inverse_transform(language) # find the language corresponding with the predicted value

    output = corr_language[0]
    return output

@app.route('/',methods=['POST','GET'])
def home():
    if request.method == 'POST':
        txt = request.form['text']
        txt  = detect(txt)
        txt = languages.get(alpha2=txt)
        res = 'Language is in {}'.format(txt.name)
        return render_template('home.html', prediction=res)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="localhost", port=8085, debug=True)


