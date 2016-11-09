#!/usr/bin/python3

from joke_collection import JokeCollection

import nltk
import pymongo
from sshtunnel import SSHTunnelForwarder

import argparse
import contextlib

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("--ssh", help="while script is running, establish ssh connection, using the "
	"parameters stored in ssh.txt", action="store_true")
parser.add_argument("--port", help="connect to your Mongo database through the provided local port", type=int, required=True)
args = parser.parse_args()

connection_string = "mongodb://localhost:{}".format(args.port)

num_jokes = 2000
top_n_terms = 10


with contextlib.ExitStack() as stack: # gives the ability to use conditional context managers
	if args.ssh:
		# get ssh connection parameters
		with open("ssh.txt", "r") as f:
			host = f.readline().strip()
			user = f.readline().strip()
			pwd = f.readline().strip()
		if args.verbose: print("opening ssh connection to {}".format(host))
		# establish ssh connection
		ssh_conn = stack.enter_context(SSHTunnelForwarder(
			host,
			ssh_username=user,
			ssh_password=pwd,
			local_bind_address=("0.0.0.0", args.port),
			remote_bind_address=("127.0.0.1", 27017)))
		if args.verbose: print("opened ssh connection")

	# establish mongo connection
	client = stack.enter_context(pymongo.MongoClient(connection_string))
	print("connected to {}".format(connection_string))
	db = client.hgp_jokerz
	collection = db.JokesCleaned

	jokes = collection.find().limit(num_jokes)
	jokes_collection = JokeCollection(jokes)
	# for category, terms in jokes_collection.max_tf_idf_by_category(n=top_n_terms, debug=args.verbose).items():
	# 	print("{}: {}".format(category, terms))
	jokes_collection.test_classifier(nltk.NaiveBayesClassifier, jokes_collection.BOW_feature_extractor, debug=args.verbose)
