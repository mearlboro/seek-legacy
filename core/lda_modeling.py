import logging
from gensim import corpora, models, similarities
import os
import glob as santa

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
files = santa.glob(os.getcwd() + "/*.txt")
# documents = santa.glob(os.getcwd() + "/txt/*.txt")

documents = []
for f in files:
    open_f = open(f, 'r')
    for line in open_f:
        documents.append(line)

stoplist = set('for a on , her she hers by which as is was were out or this that those these while for from since of the and to in'.split())

texts = [[word for word in document.lower().split() if word not in stoplist]
        for document in documents]

from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
 for token in text:
     frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
      for text in texts]
from pprint import pprint   # pretty-printer
# pprint(texts)

dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester.dict') # store the dictionary, for future reference
# print(dictionary)
# print(dictionary.token2id)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus) # store to disk, for later use
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=100) # initialize an LSI transformation
# corpus_lsi = lsi[corpus_tfidf] # c
# lsi.print_topics(300)
corpus_lda = lda[corpus_tfidf]
for doc in corpus_lda: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    print(doc)

output = open("results.txt", "w+")
output.write(str(lda.print_topics(100)))
