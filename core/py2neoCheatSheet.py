from py2neo import (authenticate, Graph, cypher, Node,
                    Relationship)
from py2neo.ext.calendar import GregorianCalendar

import time

# Returns Today's year/month/day as a graph
def todayAsGraph(calendar):
  year  = int(time.strftime("%Y"))
  month = int(time.strftime("%m"))
  day   = int(time.strftime("%d"))
  return calendar.date(year, month, day)


#Graph needed for doing anything
g = Graph("http://localhost:7474/db/data/")


# Creates a Node
#person = Node("Person", "Animal", name="Hellsing", race="werewolf")
#g.create(person)
# Creating or retrieving (Label, property, value)


#x = g.merge_one(cartujo)

# Creates or changes value of property AND stores in database
# Properties within a Node are used through dictionaries
#rula.properties["gender"] = "female" 

# Call a cypher query which returns a RecordList
# Using this for loop will iterate through the Records within the
# RecordList object returned by g.cypher.execute
for r in g.cypher.execute("MATCH (n:Animal) RETURN n"):
  # R is a record. NOT a Node. A record is a 'list' of nodes.
  print(r)




timeGraph = Graph()

calendar = GregorianCalendar(timeGraph)

x = todayAsGraph(calendar)
print(x.day)
