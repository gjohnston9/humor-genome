Requirements
============
- Python 3
- pymongo: `pip install pymongo`
- regex: `pip install regex`
- nltk: `pip install nltk`
- nltk stopwords corpus: Once you have nltk installed, run `python -c "import nltk; nltk.download()"` from your terminal, then click the "All Packages" tab at the top, scroll down to the Stopwords Corpus, and download it

Usage
=====
- Make sure ssh connection is running (`ssh -L27018:localhost:27017 [your GT name]@[our project's Mongo server]`), and make sure the connection_string in main.py is "mongodb://localhost:27018"
- Uncomment the blocks of code in main.py related to what you want to test.
    - If testing max_tf_idf_by_category, you can adjust `num_jokes` and `top_n_terms` parameters. Increasing either of these will have a significant effect on runtime.
    - If testing test_classifier, you can change the classifier used, the feature extraction method, or the number of jokes used (joke_limit parameter, with default value 5000). If using BOW_feature_extractor, you can adjust word_limit to change how many words are used (default value 2000).
- Run `python3 main.py` from your terminal.