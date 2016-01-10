# Seek: init.py
# (initial machine learning trainer)

# -----------------------------------------------------------------------------------------

"""
The code in this file instantiates (thus loads and trains) all the classes in Statistician
then dumps them as objects into pickes (python object files) for later use by the other
core modules of Seek.
The training performed here is performed on standard corpuses and most likely needs to be 
done only when it is known that corpuses have been changed/updated.
These tools can be futher trained depending on the knowledge base required by a specialised
system. To learn more see {train.py}

To access the files created by this procedure, sequentially load elements from the file:
input  = open('skills/init.pkl', 'rb')
object = pickle.load(input)
"""

# -----------------------------------------------------------------------------------------

import pickle
from datetime import datetime
from statistician import SentenceTokenizer
from statistician import McCarthyMWETokenizer
from statistician import SharoffMWETokenizer
from statistician import Collocator
from statistician import ChunkParser
from statistician import NameEntityDetector
from statistician import TopicModelling
from statistician import QuestionClassifier

sent_tok = SentenceTokenizer()
with open('skills/init_sent_tok.pkl', 'wb') as output:
    pickler = pickle.Pickler(output, -1)
    pickler.dump(sent_tok)
print(" √ " + str(datetime.now()) + ": Created trained sentence tokenizer.")
del sent_tok

mwe_dict_tok = McCarthyMWETokenizer()
with open('skills/init_mwe_dict.pkl', 'wb') as output:
    pickler = pickle.Pickler(output, -1)
    pickler.dump(mwe_dict_tok)
print(" √ " + str(datetime.now()) + ": Created multi word expression tokenizer with McCarthy dictionary.")
del mwe_dict_tok

mwe_stat_tok = SharoffMWETokenizer()
with open('skills/init_mwe_stat.pkl', 'wb') as output:
    pickler = pickle.Pickler(output, -1)
    pickler.dump(mwe_stat_tok)
print(" √ " + str(datetime.now()) + ": Created multi word expression tokenizer with Sharoff dictionary.")
del mwe_stat_tok

chunker = ChunkParser()
with open('skills/init_chunk.pkl', 'wb') as output:
    pickler = pickle.Pickler(output, -1)
    pickler.dump(chunker)
print(" √ " + str(datetime.now()) + ": Created trained chunker.")
del chunker

ner = NameEntityDetector()
with open('skills/init_ner.pkl', 'wb') as output:
    pickler = pickle.Pickler(output, -1)
    pickler.dump(ner)
print(" √ " + str(datetime.now()) + ": Created trained named entity recognizer.")
del ner

tm = TopicModelling()
with open('skills/init_topics.pkl', 'wb') as output:
    pickler = pickle.Pickler(output, -1)
    pickler.dump(tm)
print(" √ " + str(datetime.now()) + ": Created trained topic model on the Wikipedia Corpus.")
del tm 

qc = QuestionClassifier()
with open('skills/init_qc.pkl', 'wb') as output:
    pickler = pickle.Pickler(output, -1)
    pickler.dump(qc)
print(" √ " + str(datetime.now()) + ": Created trained question classifier on the QC Corpus.")
del qc


