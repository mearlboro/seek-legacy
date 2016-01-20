
import nltk
import gensim
import os
import glob
import sys
import numpy
import linguist
from py2neo import (Graph, cypher, Node, Relationship)

"""
This class is meant to analyse questions given as input in natural language and
produce a Cypher query tailored for the specific database produced by our models.
Details on Cypher and NEO4J (our GraphDB database model) can be found here:
http://neo4j.com/docs/2.3.1/cypher-getting-started.html

Questions are treated as one of the folowing types:
WHO     - AGENT/SUBJECT, PERSON
WHAT    - OBJECT/ATTRIBUTE
WHEN    - TIME
WHERE   - LOCATION
WHY     - MOTIVE (this will be a lot harder to accomplish, we need serious research here!!!)
HOW     - METHOD (same ^)

And we choose the heuristics of creating querries based on these types,
as seen above, respectively. This is doable because GraphDB allows us to assign
labels to each node, and then we can cascade through the relations between these
nodes based on the specifications of the question.


EXAMPLE:
Input information : "The bear is in the pantry. The bearpup is with the bear."
Required information: "Where is the bearpup?"

Ideally, we want our database to store something along the lines of
    (agent:BEAR)-[IS_IN]->(location:PANTRY)
    (agent:BEARPUP)-[IS_NEAR]-(agent:BEAR)

Since we have a WHERE question, we want the analyser to search for a location
(as defined previously). Thus, our analyser will iterate through all the information
it can get (ideally with with semantic restrictions, i.e. proximity and location are related)

while (result.type != location)
    find()
"""



# Assume questions start with 'wh..' word
#def getType(question):
#    type = question.split(' ')[0].lower()
 #   return {
#        'who'  : 'person'
#        'what' : 'object'
#        'when' : 'time'
#        'where': 'location'
#    }[type]


# Retrieves natural language information from src file
# and updates the database with the information  
def updatedatabase(src, graph):
        print(linguist.getrelationships(src, 0))
        
        relations = linguist.getrelationships(src, 0)
        
        for sentence in relations:
                print('HEEEELOOOOOOO ')
                print(sentence) 
                (subj, rel, ne_labels) = sentence
                (ne, label) = subj # ne is the entity's name here
                subject = Node(label, name=ne)
                g.merge_one(subject)
                if(len(ne_labels) == 1):
                        (ne, label) = ne_labels[0]
                        complement = Node(label, name=ne)
                        g.merge_one(complement)
                        relation = Relationship(subject, rel.upper(), complement)
                if(len(ne_labels) == 2):
                        (ne, label) = ne_labels[0]
                        reltype = ne
                        (ne, label) = ne_labels[1]
                        complement = Node(label, name=ne)
                        g.merge_one(complement)
                        relation = Relationship(subject, rel.upper(), complement, type=reltype)
                g.create(relation)
        
        p = graph.cypher.execute("MATCH (n : PERSON) RETURN n.name AS name")
        for n in p:
                print(n.name)
     
    

if (len(sys.argv) < 1):
        print('source file must be specified')
else:
        print('Analyser started retriving information to be stored')
        print(linguist.getrelationships(sys.argv[1], 0))
        #g = Graph("http://localhost:7474/db/data/")
        #g.schema.create_uniqueness_constraint("PERSON", "name")
        #g.schema.create_uniqueness_constraint("LOCATION", "name")
 #       updatedatabase(sys.argv[1], g)

#if  (len(sys.argv) < 3):
#  print("The analyser expects the following command \n analyse.py <command> <question>")

# A query is given. We need to analyse the query and return a result
#if (sys.argv[1] == "-a"):
#  print("TODO: question-to-answer")

