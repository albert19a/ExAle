#!/usr/bin/python

# Python Installation Resources:

# https://docs.anaconda.com/anaconda/install/windows/
# https://www.jcchouinard.com/install-python-with-anaconda-on-windows/
# https://phoenixnap.com/kb/how-to-install-python-3-windows
# https://www.ics.uci.edu/~pattis/common/handouts/pythoneclipsejava/python.html


# Mathematics and Big Data - Natural Language Processing - Practice 4
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


# Polarity classification - Opinion Mining - Sentiment Analysis

import nltk
import pandas as pd 
import wordcloud

nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Export group chat of whatsapp from some of your group
# Create a corpus of documents with this chat and do the following Sentiment Analysis of chat messages

# Define sentiment

def sentiment(document):
    vader_analyzer = SentimentIntensityAnalyzer()
    score =vader_analyzer.polarity_scores(document)
    return score

doc1 = 'Fill in the document with chat message'
sentiment(doc1)

# Example doc = "Great place to be when you are in Bangalore."
# Result: {'compound': 0.6249, 'neg': 0.0, 'neu': 0.661, 'pos': 0.339}

# Define Opinion on the basis of criteria 0.3

def opinion(text):
    vader_analyzer = SentimentIntensityAnalyzer()
    output =vader_analyzer.polarity_scores(text)

    if output['neg']>0.3:
        return 0,output['neg']
    elif  output['pos']>0.3:
        return 1,output['pos']
    return 2,output['neu']

# Review several messages to do polarity classification

doc2 = 'Fill in the document with chat message'
opinion(doc2)

# Sentiment Analyses of whole corpus:

Corpus = ["Great place to be when you are in Bangalore.",
"The place was being renovated when I visited so the seating was limited.",
"Loved the ambience, loved the foo",
"The food is delicious but not over the top.",
"Service - Little slow, probably because too many people.",
"The place is not easy to locate",
"Mushroom fried rice was tasty."]

sia = SentimentIntensityAnalyzer()

for document in Corpus:
    print(document)
    ss = sia.polarity_scores(document)
    for k in ss:
        print('{0}: {1}, '.format(k, ss[k]))

# You can also Download the dataset from https://github.com/nltk/nltk/wiki/Sentiment-Analysis

# To download : nltk.download('movie_reviews')

# Lets do sentiment analyses for movie review dataset from nltk library

# IMDB reviews SA

from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
import string

# List of words to remove
words_to_remove = stopwords.words('english') + list(string.punctuation)

#all_words = movie_reviews.words()

#List of words to keep
filtered_words = [word for word in movie_reviews.words() if not word in words_to_remove]
print(len(filtered_words)/1e6)

#Print out the most common words from filtered words

from collections import Counter
word_counter = Counter(filtered_words)
common_words = word_counter.most_common()[:10]
common_words

# Sentiment Analysis for reviews
def build_bag_of_words_features(words):
    return {word:1 for word in words if not word in words_to_remove}

positive_reviews = movie_reviews.fileids('pos')
negative_reviews = movie_reviews.fileids('neg')
negative_features = [ (build_bag_of_words_features(movie_reviews.words(fileids = [f])), 'neg')
                   for f in negative_reviews]
positive_features = [ (build_bag_of_words_features(movie_reviews.words(fileids = [f])), 'pos')
                   for f in positive_reviews]
print(len(negative_features))
print(len(positive_features))

# To better analyze the data we can convert the arrays to the pandas DataFrames; one for each class of review.

# getting data frames from the arrays
i=0
index=[]
for i in range(1000):
    index.append(i)
pos_df=pd.DataFrame(index=index,columns=['Review','Sentiment'])
neg_df=pd.DataFrame(index=index,columns=['Review','Sentiment'])

i=0
for i  in range(1000):
    rev=pos_rev[i]
    pos_df.loc[i,'Review']=rev
    pos_df.loc[i,'Sentiment']=pos_label[i]

i=0   
for i  in range(1000):
    rev=neg_rev[i]
    neg_df.loc[i,'Review']=rev
    neg_df.loc[i,'Sentiment']=neg_label[i]

pos_df.head(5) #DataFrame of positive reviews

neg_df.head(5) # DataFrame of negative reviews

#To visualize the words in the (+) and (-) reviews we can use the wordcloud module as shown:

from wordcloud import WordCloud
import matplotlib.pyplot as plt
#stop-words
stop_words=set(nltk.corpus.stopwords.words('english'))

# Generate a word cloud image for positive reviews

pos_str=''
for string in pos_df['Review']:
    if(string not in stop_words):
        pos_str=pos_str+string+' '
polarity_pos_wordcloud = WordCloud(width=600, height=400).generate(pos_str)
plt.figure( figsize=(9,9))
plt.imshow(polarity_pos_wordcloud)
plt.axis("off")
plt.tight_layout()
plt.show()

# Generate a word cloud image for negative reviews

neg_str=''
for string in neg_df['Review']:
    if(string not in stop_words):
        neg_str=neg_str+string+' '
polarity_neg_wordcloud = WordCloud(width=600, height=400).generate(neg_str)
plt.figure( figsize=(9,9))
plt.imshow(polarity_neg_wordcloud)
plt.axis("off")
plt.tight_layout()
plt.show()

#Now to do better analysis further we can combine the positive and negative DataFrames.

df=pd.concat([pos_df,neg_df],ignore_index=True) 

print(df.shape)
df.head(10)  # first 10 positive reviews

print(df.shape)
df.tail(10)  # last 10 negative reviews




