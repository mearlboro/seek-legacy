import pymongo
from pymongo import MongoClient

# Get client and database 

client       = MongoClient() # Client is the curret connection to the database

db           = client.practice # a database within the client. There may be arbitrarily
			       # many databases running (and being queried) at any time

users_col    = db.users        # collection (table-ish in RDBMS terms) within the practice database

# inserting a single element

post = {"likes": 200,"name": "emilio"}

post_id = users_col.insert_one(post).inserted_id # returns the ObjectId() field of the new entry

# retrieving single element

document = users_col.find_one({"name":"emilio"}) # find_one() can be used on any attribute
	     			                 # will find the first one according to internal ordering
				                 # orderings can be defined by calling .ensure_index(KEY: {-1, 1})
				                 # where KEY is an attribute (i.e name). 1 for ascending, -1 for descending

# inserting (bulk) more than one element


multi_post = [ {"name": "dani", "likes": 999},
		{"name": "sara", "likes": 666}]

result     = users_col.insert_many(multi_post)
result.inserted_id                              # Yields an array [ObjectId, ObjectId]

# querying multiple elements

   # Can restrict the search by passing parameters (i.e users.find({"likes":{$gt:100}}) )
for user in users_col.find().sort("likes".sort("likes")
  # user...

# removing elements 

users_col.remove({"name":"sara"}) #Removes ALL that match this criterion

# counting
users_col.count() # Hmmm, so simple. Who would've thought?