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

# browsing categories
# types = [q[2] for q in qs.values()]
# types_string = [t[0]+' '+t[1] for t in types]
# set = sorted(set(types_string))
# adding the NEs
# from statistician import NameEntityDetector
# ner = NameEntityDetector()
# f = open('../corpora/qc_selected.json','r+')
# qs = json.load(f)
# qs_nes_list = []
# for k,v in qs.items():
#     qs_nes_list += [[k, [v[0],ner.text2ne(k),v[2]]] ]
