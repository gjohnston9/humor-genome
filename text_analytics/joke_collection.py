from collections import Counter
import errno
from heapq import nlargest
from math import log
import os
import shutil
import sys


class JokeCollection:
	"""
	based on nltk.text.TextCollection 
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

		for joke in self._jokes:
			if joke["categories"] == None:
				joke["categories"] = ""
			else:
				# transform comma-separated string to array
				# TODO: modify jokes in database so that categories is an array rather than a string
				joke["categories"] = joke["categories"].split(",")

			# to avoid repeatedly calling joke.count(term)
			# TODO: would using nltk tokenization change anything here?
			joke["word_counts"] = Counter(joke["content"].lower().split())

		self._categories = set.union(*(set(joke["categories"])
			for joke in self._jokes)) # set of all distinct categories of jokes in self._jokes


	def idf(self, term):
		term = term.lower()
		idf = self._idf_cache.get(term)
		if idf is None:
			matches = sum((joke["word_counts"][term] > 0 for joke in self._jokes))
			idf = (log(len(self._jokes) / matches) if matches else 0.0)
			self._idf_cache[term] = idf
		return idf


	def max_tf_idf_by_category(self, n=10):
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
			all_jokes = " ".join(joke["content"] for joke in self._jokes if category in joke["categories"]).lower()
			words_counter = Counter(all_jokes.split())
			words = filter(lambda word: word.isalpha(), set(all_jokes.split()))
			ret[category] = nlargest(n, words, key=lambda word: words_counter[word] * self.idf(word))
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
			print "{}: directory exists. overwriting."
			shutil.rmtree(directory)
			os.makedirs(directory)
		else:
			raise Exception("{}: directory already exists.".format(directory))

		for category in self._categories:
			filename = category.replace(" ", "_").replace("/", "_") + ".txt"
			with open(os.path.join(directory, filename), "w") as f:
				f.write("\n\n~~~~\n\n".join(joke["content"].encode("utf-8") for joke in self.get_jokes(category)))
