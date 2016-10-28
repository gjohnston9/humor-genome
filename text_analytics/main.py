from joke_collection import JokeCollection

import pymongo

connection_string = "mongodb://localhost:27018"

with pymongo.MongoClient(connection_string) as client:
	db = client.hgp_jokerz
	collection = db.JokesCleaned

	jokes = collection.find().limit(1000)
	jokes_collection = JokeCollection(jokes)
	# jokes_collection.write_jokes("first_1000_jokes")
	print jokes_collection.max_tf_idf_by_category(n=10)
