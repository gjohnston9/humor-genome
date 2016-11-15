import nltk
import regex as re
import sklearn.feature_extraction.text

from collections import Counter
import errno
from heapq import nlargest
import itertools
from math import log, sqrt
import os
import pdb
import random
import shutil
import sys
import string

class JokeCollection:
	"""
	based on nltk.text.TextCollection.
	has methods for analysis of jokes/categories, and for display of jokes/categories.
	in the methods for analysis, it's important not to transform the actual self._jokes variable (i.e. removing
		punctuation, stop words, changing to lower case) because of the methods that display the jokes.
	TODO: maybe split this functionality into two different classes?
	"""
	def __init__(self, jokes):
		"""
		jokes argument should be an iterable containing jokes, where each joke is a dictionary with attributes:
			_id
			title
			content
			categories
			upvotes
			downvotes
		"""
		self._jokes = tuple(jokes) # in case a generator is passed in
		self._idf_cache = {}
		self._words = set()

		for joke in self._jokes:
			if joke["categories"] == None:
				joke["categories"] = ""
			else:
				# transform comma-separated string to array
				# TODO: modify jokes in database so that categories is an array rather than a string
				joke["categories"] = joke["categories"].split(",")

			joke_words = self.remove_punctuation(joke["content"]).lower().split()
			# to avoid repeatedly calling joke.count(term)
			joke["word_counts"] = Counter(joke_words)
			self._words.update(joke_words)

		# this will be a list in (essentially) random order, containing no duplicates,
		# of all words across all jokes in the collection
		self._words = list(self._words)

		self._categories = {}
		for joke in self._jokes:
			for category in joke["categories"]:
				self._categories[category] = self._categories.get(category, 0) + 1


	def get_all_featuresets(self, joke_limit, feature_extractor, **kwargs):
		"""
		get featureset for every joke in the collection (limited by joke_limit parameter)

		parameters:
			feature_extractor - function that will return the features of an individual joke
			**kwargs 		  - use this to pass additional arguments to feature_extractor
		"""
		jokes_to_use = random.sample(self._jokes, joke_limit)
		for joke in jokes_to_use:
			if joke["categories"] == "": # skip jokes with no category
				continue
			preprocessed_joke = self.remove_punctuation(joke["content"]).lower().split()
			yield (feature_extractor(preprocessed_joke, **kwargs), joke["categories"][0])


	def BOW_feature_extractor(self, preprocessed_joke, word_limit=2000):
		"""
		return the bag-of-words featureset for a single joke
		"""
		features = {}
		# don't use random.sample so that the words used will be consistent across all jokes
		for word in self._words[:word_limit]:
			# added .encode("utf-8") to deal with UnicodeEncodeError
			features["contains({})".format(word.encode("utf-8"))] = (word in preprocessed_joke)
		return features


	def test_classifiers(self, classifier_types, feature_extractor, joke_limit=5000, train_test_split=0.8, debug=False, **kwargs):
		"""
		train and test a classifier on the jokes in this collection

		parameters:
			classifier_type   - an *iterable* of classifiers to use, i.e. [nltk.NaiveBayesClassifier]
			joke_limit 		  - number of jokes to use
			train_test_split  - percentage of jokes to use in training set (0 < train_test_split < 1)
			debug 			  - set this to True to print information about progress, etc. inside the function
			feature_extractor - function that will return the features of an individual joke
			**kwargs		  - use this to pass additional arguments to feature_extractor
		"""
		train_test_size = int(joke_limit * train_test_split)
		if debug: print("getting feature sets")
		featuresets = tuple(self.get_all_featuresets(joke_limit, feature_extractor, **kwargs))
		if debug: print("getting training set ({} items) and testing set ({} items)".format(
			joke_limit * train_test_split, len(featuresets) - (joke_limit * train_test_split)))
		train_set, test_set = featuresets[:train_test_size], featuresets[train_test_size:]
		for classifier_type in classifier_types:
			if debug: print("training {} classifier".format(classifier_type))
			classifier = classifier_type.train(train_set)
			if debug: print("finished training.")
			# TODO: write separate method to create train and test set		
			print("accuracy on test set: {}".format(nltk.classify.accuracy(classifier, test_set)))


	@staticmethod
	def remove_punctuation(text):
		"""
		remove punctuation from unicode string
		"""
		return re.sub(r"\p{P}+", " ", text) # contained in stopwords list are terms like "wouldn", so replacing
		# punctuation with spaces allows for removal of things like this


	def idf(self, term):
		term = term.lower()
		idf = self._idf_cache.get(term)
		if idf is None:
			matches = sum((joke["word_counts"][term] > 0 for joke in self._jokes))
			idf = (log(len(self._jokes) / matches) if matches else 0.0)
			self._idf_cache[term] = idf
		return idf


	def max_tf_idf_by_category(self, n=10, debug=False):
		"""
		return a dictionary mapping each category in the collection to the words that
		1) frequently appear in jokes in this category, and
		2) do not frequently appear in jokes in other categories
		"""
		# TODO: remove stop words
		ret = {}
		for category in self._categories:
			if category == "":
				continue
			if debug:
				print("beginning max_tf_idf_by_category for {}".format(category))

			# use of stopwords really shouldn't be necessary since we're using a version of tf-idf
			# TODO: look at results of 1) using more jokes, or 2) changing idf weighting
			# https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Inverse_document_frequency_2
			stopwords_list = nltk.corpus.stopwords.words("english")

			words_counter = Counter() # maps each word to the sum of the square roots of the number of occurrences
			# of the word in each joke in this category
			all_words = set() # set of all words in jokes in this category

			for joke in self.get_jokes(category):
				content = self.remove_punctuation(joke["content"].lower()).split()
				all_words.update(filter(lambda word: word.isalpha() and word not in stopwords_list, content)) # add any words not seen yet
				counts = Counter(content)
				words_counter += Counter({word : sqrt(counts[word]) for word in counts})

			ret[category] = nlargest(n, all_words, key=lambda word: words_counter[word] * self.idf(word))
		return ret


	def get_jokes(self, category):
		"""
		get all jokes in the collection belonging to the specified category
		"""
		return (joke for joke in self._jokes if category in joke["categories"])


	def write_jokes(self, directory, overwrite=False):
		"""
		- create a directory with the specified name (if such a directory already exists, raise an exception if
		overwrite argument is False, otherwise erase the directory and then create it)
		- then for each category, create a text file in the directory, containing all jokes
		belonging to that category (there will be overlap between files since there are many jokes that belong
		to multiple categories)
		"""
		if not os.path.exists(directory):
			os.makedirs(directory)
		elif overwrite:
			print("{}: directory exists. overwriting.".format(directory))
			shutil.rmtree(directory)
			os.makedirs(directory)
		else:
			raise Exception("{}: directory already exists.".format(directory))

		for category in self._categories:
			filename = category.replace(" ", "_").replace("/", "_") + ".txt"
			with open(os.path.join(directory, filename), "w") as f:
				# first line contains number of jokes that belong to this category
				f.write(str(self._categories[category]))
				f.write("\n\n~~~~\n\n".join(joke["content"].encode("utf-8") for joke in self.get_jokes(category)))
		print("{}: finished writing".format(directory))
