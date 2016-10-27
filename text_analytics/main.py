from joke_collection import JokeCollection

import pymongo

with pymongo.MongoClient("localhost", 27017) as client:
	db = client.hgp_jokerz
	collection = db.jokes

	jokes = (joke for joke in collection.find().limit(1000))
	jokes_collection = JokeCollection(jokes_list)
	# jokes_collection.write_jokes("first_1000_jokes")
	print jokes_collection.max_tf_idf_by_category(n=10)
