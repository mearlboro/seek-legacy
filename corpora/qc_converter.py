'''
quick script that generates the qc corpus with NE from the original nltk qc corpus

IMPORTANT:
to use move in /core
'''

import nltk
from statistician import NameEntityDetector

f = open('qc.txt')
lines = f.readlines()
toks = list(map(nltk.word_tokenize,lines))
tuples = [(line[3:],(line[0],line[3])) for line in toks]

ner = NameEntityDetector()


