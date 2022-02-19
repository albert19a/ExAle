#!/usr/bin/python

# Mathematics and Big Data - Text Mining Practice 2
# Universitat Autonoma de barcelona
# All rights reserved.

# Importing the Python libraries

# To install a Python Package

# pip install package_to_install
import sklearn
import os
import scipy
import numpy as np
import matplotlib
import wordcloud
from wordcloud import WordCloud
import string
import math
import string
import sys
import pandas as pd
import sklearn.feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# Document term frequency

# Generting documents
doc_1 = "John and Bob are brothers, John is elder than Bob"
doc_2 = "John went to the store"
doc_3 = "Bob went to the store"

corpus = [doc_1, doc_2, doc_3]

def tf(corpus):
    tfs = []
    for document in corpus:
        dic={}
        for word in document.split():
            if word in dic:
                dic[word]+=1
            else:
                dic[word]=1
        for word,freq in dic.items():
            print(word,freq)
            dic[word]=freq/len(document.split())
        tfs.append(dic)
    return tfs

tf(corpus)

#Output

# ('Indonesia,', 1)
# ('and', 1)
# ('United', 1)
# ('Brazil.', 1)
# ('countries', 1)
# ('States,', 1)
# ('India,', 1)
# ('5', 1)
# ('are', 1)
# ('in', 1)
# ('biggest', 1)
# ('2020', 1)
# ('The', 1)
# ('China,', 1)
# ('by', 1)
# ('population', 1)
# ('A', 1)
# ('and', 2)
# ('Boul', 2)
# ('B', 1)
# ('contains', 2)
# ('3', 1)
# ('blue', 1)
# ('while', 1)
# ('4', 2)
# ('6', 1)
# ('balls,', 1)
# ('white', 1)
# ('balls.', 1)
# ('red', 2)
# ('string.', 1)
# ('&is', 1)
# ('[an]', 1)
# ('This', 1)
# ('with.?', 1)
# ('{of}', 1)
# ('example?', 1)
# ('punctuation!!!!', 1)


# Document Term Matrix

# Finding TDM via textmining package
import textmining 
tdm = textmining.TermDocumentMatrix()

# Creating TDM for each document

tdm.add_doc(doc_1)
tdm.add_doc(doc_2)
tdm.add_doc(doc_3)

for row in tdm.rows(cutoff=1):
    print(row)

# Output TDM for doc_1, doc_2, doc_3

#['and', 'the', 'brothers', 'to', 'are', 'bob', 'john', 'went', 'store']
#[1, 0, 1, 0, 1, 1, 1, 0, 0]
#[0, 1, 0, 1, 0, 0, 1, 1, 1]
#[0, 1, 0, 1, 0, 1, 0, 1, 1]

# Writing TDM in csv file

tdm.write_csv('matrix.csv', cutoff = 1)

# Finding TDM by using sklearn Package

corpus = [doc_1, doc_2, doc_3]

vec = CountVectorizer()
X = vec.fit_transform(corpus)
df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
print(df)

# and  are  bob  brothers  john  store  the  to  went
#0    1    1    1         1     1      0    0   0     0
#1    0    0    0         0     1      1    1   1     1
#2    0    0    1         0     0      1    1   1     1

# Bag of words

bow = vec.get_feature_names()
print(bow)

# or
bow = list(count_vectorizer.vocabulary_.items())
print(bow[:10])


############# Text Visualization ##############

# Start with one document:

document = "Write or read your document from corpus here"

##### WordCloud  #######

# Create and generate a word cloud image:

wordcloud = WordCloud().generate(document)

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Choose font and background options in function visualize as follows:

def visualize(text):
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# Save the image in the img folder:

wordcloud.to_file("first_document.png")

# To visualize the word cloud for text
visualize(document)



######### SVM Text Classifier ############

# The objective of the support vector machine algorithm is to find a hyperplane 
# in an N dimensional space(N the number of features) that distinctly classifies 
# the data points.


# Create a Counter of tokens
count_vectorizer = CountVectorizer(decode_error='ignore', lowercase=True, min_df=2)
# Apply it on the train data to get the vocabulary and the mapping. 
#This vocab and mapping is then applied to the test set.
# Before, we convert to Unicode to avoid issues with CountVectorizer
# Divide the data in to Test and Train data
train = count_vectorizer.fit_transform(X_train.values.astype('U'))
test = count_vectorizer.transform(X_test.values.astype('U'))

print('Train size: ',train.shape)
print('Test size: ',test.shape)

# Extract the vocabulary as a list of (word, frequency)
vocab = list(count_vectorizer.vocabulary_.items())
print(vocab[:10])


# Define the parameters to tune
parameters = { 
    'C': [1.0, 10],
    'gamma': [1, 'auto', 'scale']
}
# Tune yyperparameters  using Grid Search and a SVM model
model = GridSearchCV(SVC(kernel='rbf'), parameters, cv=5, n_jobs=-1).fit(train, y_train)

# Predicting the Test set results
y_pred = model.predict(test)

print(metrics.classification_report(y_test, y_pred,  digits=5))
plot_confussion_matrix(y_test, y_pred)
plot_roc_curve(y_test, y_pred)

 
############ Document Similarity ################

# Cosine distance word embedding method

def cos_distance(doc_1, doc_2):   
    vec_1 = np.mean([model[word] for word in preprocess(doc_1)],axis=0)
    vec_2 = np.mean([model[word] for word in preprocess(doc_2)],axis=0)
    cos = scipy.spatial.distance.cosine(vec_1, vec_2)
    print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cos)*100,2),'%')


# Calculate document similarity with Angle Based Matching Method.
  
# Reading the text file This functio will return a 
# list of the lines of text in the file.

def read_document(document): 
      
    try:
        with open(document, 'r') as f:
            data = f.read()
        return data
      
    except IOError:
        print("Error opening or reading input file: ", document)
        sys.exit()
  
# splitting the text lines into words
# translation table is a global variable
# mapping upper case to lower case and
# punctuation to spaces
translation_table = str.maketrans(string.punctuation+string.ascii_uppercase,
                                     " "*len(string.punctuation)+string.ascii_lowercase)
       
# returns a list of the words in the file from line list

def get_words(text): 
      
    text = text.translate(translation_table)
    word_list = text.split()
      
    return word_list
  
  
# counts frequency of each word returns a dictionary which maps
# the words to  their frequency.

def count_frequency(word_list): 
      
    D = {}
      
    for new_word in word_list:
          
        if new_word in D:
            D[new_word] = D[new_word] + 1
              
        else:
            D[new_word] = 1
              
    return D
  
# returns dictionary of (word, frequency) pairs from the previous dictionary.

def word_frequencies_for_file(document): 
      
    line_list = read_file(document)
    word_list = get_words_from_line_list(line_list)
    freq_mapping = count_frequency(word_list)
  
    print("File", document, ":", )
    print(len(line_list), "lines, ", )
    print(len(word_list), "words, ", )
    print(len(freq_mapping), "distinct words")
  
    return freq_mapping
  
  
# returns the dot product of two documents

def dotProduct(D1, D2): 
    Sum = 0.0
      
    for key in D1:
          
        if key in D2:
            Sum += (D1[key] * D2[key])
              
    return Sum
  
# returns the angle in radians between document vectors

def vector_angle(D1, D2): 
    numerator = dotProduct(D1, D2)
    denominator = math.sqrt(dotProduct(D1, D1)*dotProduct(D2, D2))
      
    return math.acos(numerator / denominator)
  
  
def document_similarity(document_1, document_2):
      
   # document_1 = sys.argv[1]
   # document_2 = sys.argv[2]
    sorted_word_list_1 = word_frequencies_for_file(document_1)
    sorted_word_list_2 = word_frequencies_for_file(document_2)
    distance = vector_angle(sorted_word_list_1, sorted_word_list_2)
      
    print("The distance between the documents is: % 0.6f (radians)"% distance)
      

