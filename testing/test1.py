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

documents['title'] = documents['title'].fillna('999')
documents = documents.drop(documents[(documents.title=='999')].index.tolist())

documents = documents.drop(documents[(documents.title=='Cours de EFPC / CSPS courses')].index.tolist())

documents = documents.drop(documents[(documents.title=='Cours provenant d\'une autre ministère / Other Government Department courses')].index.tolist())

documents = documents.drop(documents[(documents.title=='Externals Ressources Links')].index.tolist())

documents = documents.drop(documents[(documents.title=='Cours de Skillsoft courses')].index.tolist())

documents = documents.drop(documents[(documents.title=='Delegation of Authority - video')].index.tolist())

documents = documents.drop(documents[(documents.title=='Blogs')].index.tolist())

documents = documents.drop(documents[(documents.title=='Cases Studies')].index.tolist())

documents = documents.drop(documents[(documents.title=='hq')].index.tolist())

documents = documents.drop(documents[(documents.title=='Video List')].index.tolist())


# stemmer = SnowballStemmer("english")
# def lemmatize_stemming(text):
#     return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
# def preprocess(text):
#     result = []
#     for token in gensim.utils.simple_preprocess(text):
#         if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
#             result.append(lemmatize_stemming(token))
#     return result


# doc_sample = documents[documents.index == 4].values[0][0]
# print('\noriginal document: ')
# words = []
# for word in doc_sample.split(' '):
#     words.append(word)
# print(words)
# print('\ntokenized and lemmatized document: ')
# print(preprocess(doc_sample))
# processed_docs = documents['title'].map(preprocess)
# processed_docs = processed_docs[:700]


# # Bag of words on the dataset
# dictionary = gensim.corpora.Dictionary(processed_docs)

# count = 0
# for k, v in dictionary.iteritems():
#     print(k, v)
#     count += 1
#     if count > 10:
#         break
# bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=3)


# from gensim.test.utils import datapath
# # temp_file = datapath("model")
# # lda_model.save(temp_file)
# lda_model.save("word2vec.model")

# print("finished")