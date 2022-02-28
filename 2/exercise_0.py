import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import re
import string
from pprint import pprint
import pandas as pd
import math 

import requests
import urllib
from requests_html import HTML
from requests_html import HTMLSession

import gensim
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import operator

# collecting Google search results for the "how to build a website" query
def get_source(url):
    try:
        session = HTMLSession()
        response = session.get(url)
        return response

    except requests.exceptions.RequestException as e:
        print(e)

def get_results(query):
    query = urllib.parse.quote_plus(query)
    response = get_source("https://www.google.com/search?q=" + query)
    
    return response

def parse_results(response):
    css_identifier_result = ".tF2Cxc"
    css_identifier_title = "h3"
    results = response.html.find(css_identifier_result)
    
    output = []
    for result in results:
        output.append(result.find(css_identifier_title, first=True).text)
    return output

def google_search(query):
    response = get_results(query)
    return parse_results(response)

results = google_search("how to build a website")
corpus = results
pprint(corpus)

# text preprocessing
def is_str_nan(str):
    '''
    Returns True if the value is nan
    '''
    return str != str

def m_freq(text,n_words=2):
    '''
    Returns the most (2 by default) frequently used words from a text
    ''' 
    fdist = FreqDist(word.lower() for word in word_tokenize(text))
    words = fdist.most_common(n_words)
    ret = []
    for word in words:
        ret.append(word[0])
    print("frequent words: ",ret)        
    return ret

def m_rare(text,n_words=2):
    '''
    Returns the least (2 by default) frequently used words from a text
    ''' 
    fdist = FreqDist(word.lower() for word in word_tokenize(text))
    words = fdist.most_common()[-n_words:]
    ret = []
    for word in words:
        ret.append(word[0])
    print("rare words: ",ret)
    return ret

def text_cleaner(corpus):
    """ 
    CHECKS for capitalization
    REMOVES punctuation, stop words, nan
    """
    ret = []
    all_stopwords = gensim.parsing.preprocessing.STOPWORDS
    punct =  set(string.punctuation)
    to_remove = all_stopwords.union(punct)
    for document in corpus:
        words = nltk.word_tokenize(document)
        doc = [word.lower() for word in words if word.isalnum() if is_str_nan(word) != True if word not in to_remove]
        filtered_doc = " ".join(doc)
        ret.append(filtered_doc)
    return ret
    

pprint(text_cleaner(corpus))

def text_cleaner_frequency(corpus):
    """ 
    REMOVES rare words, frequent words from a corpus of documents
    """
    ret = text_cleaner(corpus)
    text = " ".join(ret)
    most_frequent = m_freq(text)
    most_rare = m_rare(text)
    to_remove = set(most_frequent)
    to_remove = to_remove.union(set(most_rare))
    r = []
    for document in ret:
        words = nltk.word_tokenize(document)
        doc = [word for word in words if word not in to_remove]
        filtered_doc = " ".join(doc)
        r.append(filtered_doc)
    return r
print("cleaned text: ", text_cleaner_frequency(corpus))