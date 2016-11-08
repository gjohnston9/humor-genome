from joke_collection import JokeCollection

import nltk
import pymongo

connection_string = "mongodb://localhost:27018"
num_jokes = 10000
top_n_terms = 10

with pymongo.MongoClient(connection_string) as client:
	print("connected to {}".format(connection_string))
	db = client.hgp_jokerz
	collection = db.JokesCleaned

	jokes = collection.find().limit(num_jokes)
	jokes_collection = JokeCollection(jokes)
	# for category, terms in jokes_collection.max_tf_idf_by_category(n=top_n_terms, debug=True).items():
	# 	print "{}: {}".format(category, terms)
	jokes_collection.test_classifier(nltk.NaiveBayesClassifier, debug=True)
