import numpy
import nltk
import gensim
import os
import glob
import sys
import numpy

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
def getType(question):
    type = question.split(' ')[0].lower()
    return {
        'who'  : 'person'
        'what' : 'object'
        'when' : 'time'
        'where': 'location'
    }[type]
