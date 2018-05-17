#import libraries 
import pandas as pd 
import numpy as np 
import nltk 
from nltk.stem import PorterStemmer, WordNetLemmatizer
import gensim 
from gensim.models import word2vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

news=pd.read_csv("egg_yolk.csv")
#show different news outlets
news['channel'].values
#subset by news outlet 
cnn=news[news['channel'].str.contains("CNNW")] #also for BBC,MSNBC 
cnn['duration'].mean() #3.38 seconds for words on CNN screen 
#word counts
cnn['text'].value_counts() 
cnn_palabras=cnn['text'].T.tolist() #list 

#tokenize 
tokenize_list=[word_tokenize(i) for i in cnn_palabras]
cnn_palabras1=' '.join(cnn_palabras) #list to string 
type(cnn_palabras1) #string 
cnn_palabras2=cnn_palabras1.lower() #string to lowercase 

#word embeddings 
model=word2vec.Word2Vec(cnn_palabras2) 
palabrasX=list(model.wv.vocab)

#train the model
model=word2vec.Word2Vec(tokenize_list,min_count=45)
model
#summarize the vocabulary 
words=list(model.wv.vocab)
#access the vector for a popular word 
model['TRUMP']

#visaulize the word embeddings 
X=model[model.wv.vocab]

#plot the word vectors using PCA 
pca=PCA(n_components=2) 
result=pca.fit_transform(X)

#scatterplot
pyplot.scatter(result[:,0],result[:,1])
words=list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word,xy=(result[i,0],result[i,1]))
pyplot.show() 

























