#! /usr/bin/python

import nltk
import json
import sys

"""
This script filters the nltk question classifier corpus. It needs to be passed a
file containing each desired types of questions, each on a new lines

e.g. ENTY
     NUM
"""

print(sys.argv[1])
file_qtypes = sys.argv[1]
f = open(file_qtypes)
qtypes = f.readlines()
qtypes = [t[:-1] for t in qtypes] #because /n is a thing
with open('qc.json') as q:
    qc = json.load(q)

questions = dict((q, qtype) for (q, qtype) in qc.items() if any(qtype[2][0] in t for t in qtypes))

with open('qc_selected.json', 'w') as out:
    json.dump(questions, out)
