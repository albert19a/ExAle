#!/usr/bin/python

# Python Installation Resources:

# https://docs.anaconda.com/anaconda/install/windows/
# https://www.jcchouinard.com/install-python-with-anaconda-on-windows/
# https://phoenixnap.com/kb/how-to-install-python-3-windows
# https://www.ics.uci.edu/~pattis/common/handouts/pythoneclipsejava/python.html


# Mathematics and Big Data - Text Mining Practice 5
# Universitat Autonoma de barcelona
# All rights reserved.

# Importing the Python libraries

# If required
#conda install nltk[twitter] or
#pip3 install -U nltk[twitter]
# How to install nltk library in python
#For Python version 2.x:
#pip install nltk
#For Python version 3.x:
#pip3 install nltk 
#For mac use sudo pip3 install nltk
# import libraries


############################################################################################################


# Topic modeling 

# Data obtained from www.npr.org

# Import Package
import pandas as pd

# Read the quora questions file
npr = pd.read_csv('npr.csv')

npr.head()

#Using TF-IDF Vectorization to create a vectorized document term matrix

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

dtm = cv.fit_transform(npr['Article'])

dtm # Document term matrix

### LDA

from sklearn.decomposition import LatentDirichletAllocation

LDA = LatentDirichletAllocation(n_components=7,random_state=42)

LDA.fit(dtm)

# Store words for topics

len(cv.get_feature_names())

import random

for i in range(10):
    random_word_id = random.randint(0,54776)
    print(cv.get_feature_names()[random_word_id])

for i in range(10):
    random_word_id = random.randint(0,54776)
    print(cv.get_feature_names()[random_word_id])

# Top words per topic

len(LDA.components_)

LDA.components_

len(LDA.components_[0])

single_topic = LDA.components_[0]

single_topic.argsort() # Return indices 

# Top 10 words for this topic

single_topic.argsort()[-10:] # Return indices

top_word_indices = single_topic.argsort()[-10:]

# Print top words 
for index in top_word_indices:
    print(cv.get_feature_names()[index])

# View first 10 topics

for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')

# Clusters of words for each topic

# THE TOP 15 WORDS FOR TOPIC #0
# ['companies', 'money', 'year', 'federal', '000', 'new', 'percent', 'government', 'company', 'million', 'care', 'people', 'health', 'said', 'says']


# THE TOP 15 WORDS FOR TOPIC #1
# ['military', 'house', 'security', 'russia', 'government', 'npr', 'reports', 'says', 'news', 'people', 'told', 'police', 'president', 'trump', 'said']


# THE TOP 15 WORDS FOR TOPIC #2
# ['way', 'world', 'family', 'home', 'day', 'time', 'water', 'city', 'new', 'years', 'food', 'just', 'people', 'like', 'says']


# THE TOP 15 WORDS FOR TOPIC #3
# ['time', 'new', 'don', 'years', 'medical', 'disease', 'patients', 'just', 'children', 'study', 'like', 'women', 'health', 'people', 'says']


# THE TOP 15 WORDS FOR TOPIC #4
# ['voters', 'vote', 'election', 'party', 'new', 'obama', 'court', 'republican', 'campaign', 'people', 'state', 'president', 'clinton', 'said', 'trump']


# THE TOP 15 WORDS FOR TOPIC #5
# ['years', 'going', 've', 'life', 'don', 'new', 'way', 'music', 'really', 'time', 'know', 'think', 'people', 'just', 'like']


# THE TOP 15 WORDS FOR TOPIC #6
# ['student', 'years', 'data', 'science', 'university', 'people', 'time', 'schools', 'just', 'education', 'new', 'like', 'students', 'school', 'says']



# Linking the topics and documents.

dtm

topic_results = LDA.transform(dtm)

topic_results[0]

topic_results[0].round(2)

topic_results[0].argmax() # The first document belings to topic 1.

# Topic retrieval for original data

npr.head()

topic_results.argmax(axis=1)

npr['Topic'] = topic_results.argmax(axis=1)

npr.head(10)

################ LSA ################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_colwidth", 200)


# Data set can be downloaded from https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups

from sklearn.datasets import fetch_20newsgroups

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
len(documents)

dataset.target_names

# Data Preprocessing
news_df = pd.DataFrame({'document':documents})

# removing everything except alphabets`
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z#]", " ")

# removing short words
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

# make all text lowercase
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())


# Removing stopwords

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# tokenization
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())

# remove stop-words
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

# de-tokenization
detokenized_doc = []
for i in range(len(news_df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

news_df['clean_doc'] = detokenized_doc

# Creating DTM

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', 
max_features= 1000, # keep top 1000 terms 
max_df = 0.5, 
smooth_idf=True)

X = vectorizer.fit_transform(news_df['clean_doc'])

X.shape # check shape of the document-term matrix

# Topic Modeling

#Data comes from 20 different newsgroups, let’s try to have 20 topics for our text data

from sklearn.decomposition import TruncatedSVD

# SVD represent documents and terms in vectors 
svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)

svd_model.fit(X)

len(svd_model.components_)

terms = vectorizer.get_feature_names()

for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    print("Topic "+str(i)+": ")
    for t in sorted_terms:
        print(t[0])
        print(" ")

#Output

# Topic 0: like know people think good time thanks
# Topic 1: thanks windows card drive mail file advance
# Topic 2: game team year games season players good
# Topic 3: drive scsi disk hard card drives problem
# Topic 4: windows file window files program using problem
# Topic 5: government chip mail space information encryption data
# Topic 6: like bike know chip sounds looks look
# Topic 7: card sale video offer monitor price jesus
# Topic 8: know card chip video government people clipper
# Topic 9: good know time bike jesus problem work
# Topic 10: think chip good thanks clipper need encryption
# Topic 11: thanks right problem good bike time window
# Topic 12: good people windows know file sale files
# Topic 13: space think know nasa problem year israel
# Topic 14: space good card people time nasa thanks
# Topic 15: people problem window time game want bike
# Topic 16: time bike right windows file need really
# Topic 17: time problem file think israel long mail
# Topic 18: file need card files problem right good
# Topic 19: problem file thanks used space chip sale






