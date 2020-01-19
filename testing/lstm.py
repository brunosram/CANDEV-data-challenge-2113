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

#label
label = data.iloc[:,63:82]
label = label.fillna(0)
label = label.replace('X', 1)
label = label.replace('x', 1)
label = label.values
print(np.shape(label))

data_text = data[['title','simple description','description for Gccampus']]
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

for i in documents.index:
    if(' '.join(str(documents['simple description'][i]).split()[:10])==' '.join(str(documents['description for Gccampus'][i]).split()[:10])):
        documents['simple description'][i]= ""

# documents.to_csv("clean.csv")

documents['combine'] = documents[['title','simple description','description for Gccampus']].apply(lambda x : '{} {} {}'.format(x[0],x[1],x[2]), axis=1)

# documents.to_csv("combine.csv")



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
# =============================================================================
# processed_docs = processed_docs[:100]
# =============================================================================


# Bag of words on the dataset6
dictionary = gensim.corpora.Dictionary(processed_docs)

count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


# =============================================================================
# lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=3)
# lda_model = gensim.models.ldamodel.LdaModel.load("word2vec.model")
# 
# for idx, topic in lda_model.print_topics(-1):
#     print('Topic: {} \nWords: {}'.format(idx, topic))
# =============================================================================
# pad and transform
padded_length = 100
length = len(bow_corpus)
padded_bow = np.zeros([length, padded_length])
for i in range(length):
    for j in range(len(bow_corpus[i])):
        padded_bow[i][2*j] = bow_corpus[i][j][0]
        padded_bow[i][2*j+1] = bow_corpus[i][j][1]