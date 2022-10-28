# -*- coding: utf-8 -*-
"""BERT for Indian Products on Amazon - Sentiment Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XZj33F75Iq9C1p8bOgOwr1gVbPe_mRfm
"""

!pip install transformers
!pip install ipywidgets
!pip install -U tensorflow-text==2.6.0

from google.colab import drive
drive.mount('/content/drive/')

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

import tensorflow_hub as hub
from tensorflow.keras.models import Model
import tensorflow_text
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Commented out IPython magic to ensure Python compatibility.
# %cd "D:\Study\Chatbot\DL_bot"
# %cd "/content/drive/My Drive/Study/Chatbot/DL_bot"

reviews_df = pd.read_csv('amazon_vfl_reviews.csv')

reviews_df.shape

reviews_df.info()

reviews_df.drop(['asin','name','date'],axis=1,inplace=True)

reviews_df.info()

reviews_df.head(2)

reviews_df.rating.value_counts()

reviews_df['sent_rating'] = reviews_df.rating.apply(lambda x: 1 if x>=4 else 0)

reviews_df.sent_rating.value_counts()

reviews_df.drop('rating',axis=1,inplace=True)

import re
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop = stopwords.words('english')

def clean_text(review_text):
  text = str(review_text).lower()
  text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
  text = " ".join([word for word in text.split() if word not in (stop)])
  text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
  return text

reviews_df['review'] = reviews_df.review.apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(reviews_df['review'],reviews_df['sent_rating'],test_size=0.3,random_state=1234,stratify=reviews_df['sent_rating'])

X_train.shape, y_train.shape, X_test.shape, y_test.shape

tf.keras.backend.clear_session()

#BERT takes 3 inputs - input word ids, input mask for padding, segment ids - for sentences

def create_model():
   
        
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

        #bert layer 
    bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

    preprocessed_text = bert_preprocess(text_input)
    output = bert_layer(preprocessed_text)

    output = tf.keras.layers.Dense(1,"sigmoid")(output['pooled_output'])
    #Bert model
    #We are using only pooled output not sequence out. 
    #If you want to know about those, please read https://www.kaggle.com/questions-and-answers/86510
    model = Model(inputs=text_input, outputs=output)

        
    return model

model = create_model()

model.summary()

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=METRICS)

model.fit(X_train, y_train, validation_data = (X_test,y_test),epochs=10)

y_pred = model.predict(X_test[:10])

y_pred[y_pred <= 0.5] = 0.
y_pred[y_pred > 0.5] = 1.

len(y_pred)

from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix(y_test[:9],y_pred)

print(classification_report(y_test[:9], y_pred))

model.save('sentiment_analyzer')

from tensorflow.keras import models

mymodel = models.load_model("sentiment_analyzer")

mymodel.predict(['This is my review. Not a great product'])

mymodel.summary()

