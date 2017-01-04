#!/usr/bin/python3

from joke_collection import JokeCollection

import json
from nltk.classify import NaiveBayesClassifier, MaxentClassifier, DecisionTreeClassifier
import pymongo
from sshtunnel import SSHTunnelForwarder

import argparse
import contextlib

parser = argparse.ArgumentParser()
parser.add_argument("--quiet", help="decrease output verbosity", action="store_true")
parser.add_argument("--ssh", help="while script is running, establish ssh connection, using the "
	"parameters stored in ssh.txt", action="store_true")
parser.add_argument("--port", help="(required) connect to your Mongo database through the provided local port", type=int, required=True)
args = parser.parse_args()

connection_string = "mongodb://localhost:{}".format(args.port)

num_jokes = 11000
top_n_terms = 10


with contextlib.ExitStack() as stack: # gives the ability to use conditional context managers
	if args.ssh:
		# get ssh connection parameters
		with open("credentials.json", "r") as f:
			data = json.load(f)
			host = data["ssh"]["host"]
			user = data["ssh"]["user"]
			pwd = data["ssh"]["password"]
		if not args.quiet: print("opening ssh connection to {}".format(host))
		# establish ssh connection
		ssh_conn = stack.enter_context(SSHTunnelForwarder(
			host,
			ssh_username=user,
			ssh_password=pwd,
			local_bind_address=("0.0.0.0", args.port),
			remote_bind_address=("127.0.0.1", 27017)))
		if not args.quiet: print("opened ssh connection")

	# establish mongo connection
	client = stack.enter_context(pymongo.MongoClient(connection_string))
	print("connected to {}".format(connection_string))
	db = client.hgp_jokerz
	collection = db.JokesCleaned

	jokes = collection.find().limit(num_jokes)
	jokes_collection = JokeCollection(jokes)

	# keywords = jokes_collection.max_tf_idf_by_category(n=top_n_terms, debug=not args.quiet)
	# for category, terms in keywords.items():
	# 	print("{}: {}".format(category, terms))

	classifier_types = [
		NaiveBayesClassifier,
		]

	# print("\n\nnltk - word count features:")
	# jokes_collection.test_classifiers(classifier_types, jokes_collection.word_count_feature_extractor, debug=not args.quiet)

	# print("\n\nnltk - bag of words (binary counts) features:")
	# jokes_collection.test_classifiers(classifier_types, jokes_collection.BOW_feature_extractor, debug=not args.quiet)

	jokes_collection.sklearn_test(debug=not args.quiet, vectorize="count", joke_limit=num_jokes-1000)

	# jokes_collection.classify_demo(5, keywords)
