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
from statistician import SentenceTokenizer

# pickler will dump the classes into object files  
output  = open('skills/init.pkl', 'wb') 
pickler = pickle.Pickler(output, -1)

# instantiate the classes

sentenceTokenizer = SentenceTokenizer()
pickler.dump(sentenceTokenizer)
del sentenceTokenizer

