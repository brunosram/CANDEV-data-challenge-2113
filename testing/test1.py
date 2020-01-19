# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:28:30 2020

@author: Yuan
"""

import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
# =============================================================================
# nltk.download('wordnet')
# =============================================================================
data = pd.read_csv('g.csv', error_bad_lines=False, encoding = "ISO-8859-1")
data_text = data[['title']]
documents = data_text
stemmer = SnowballStemmer("english")
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

documents['title'] = documents['title'].fillna('999')
documents = documents.drop(documents[(documents.title=='999')].index.tolist())

doc_sample = documents[documents.index == 4].values[0][0]
print('\noriginal document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\ntokenized and lemmatized document: ')
print(preprocess(doc_sample))
processed_docs = documents['title'].map(preprocess)
processed_docs = processed_docs[:100]


# Bag of words on the dataset
dictionary = gensim.corpora.Dictionary(processed_docs)

count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=3)

print("finished")

