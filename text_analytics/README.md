Requirements
============
- Python 3.3+
- pymongo: `pip install pymongo`
- regex: `pip install regex`
- nltk: `pip install nltk`
- sshtunnel: `pip install sshtunnel`
- nltk stopwords corpus: Once you have nltk installed, run `python -c "import nltk; nltk.download()"` from your terminal, then click the "All Packages" tab at the top, scroll down to the Stopwords Corpus, and download it.

Usage
=====
- main.py provides the option to establish an SSH connection for you (use the --ssh flag when running main.py). If using this, create a file called "ssh.txt" in the same directory as main.py, with the host address on the first line, your GT username on the second line, and your GT password on the third line. An SSH connection will be created, with port forwarding from your local port (specified by the --port argument) to 27017 at the host address.
- Uncomment the blocks of code in main.py related to what you want to test.
    - If testing `max_tf_idf_by_category`, you can adjust `num_jokes` and `top_n_terms` parameters. Increasing either of these will have a significant effect on runtime.
    - If testing `test_classifier`, you can change the classifier used, the feature extraction method, or the number of jokes used (`joke_limit` parameter, with default value 5000). If using `BOW_feature_extractor`, you can adjust word_limit to change how many words are used (default value 2000).
- Run `python3 main.py --port <port>` from your terminal, where `<port>` is the local port through which your Mongo database will be accessed.
- Additional command-line options can be viewed by running `python3 main.py -h`.