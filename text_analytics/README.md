Requirements
============
- Python 2.4 or higher
- pymongo: `pip install pymongo`
- regex: `pip install regex`
- nltk: `pip install nltk`
- nltk stopwords corpus: once you have nltk installed, run `python -c "import nltk; nltk.download()"` from your terminal, then click the "All Packages" tab at the top, scroll down to the Stopwords Corpus, and download it

Usage
=====
- make sure ssh connection is running (`ssh -L27018:localhost:27017 [your GT name]@[our project's Mongo server]`)
- adjust `num_jokes` and `top_n_terms` parameters in main.py (increasing `num_jokes` will cause the program to take significantly more time to finish)
- run `python main.py` from your terminal