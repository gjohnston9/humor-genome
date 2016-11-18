# humor-genome
- cleanup contains SQL queries that were used in cleaning up the jokes in one of our databases (removing subcategories, combining some top-level categories, moving all information into one table in order to make SQL -> Mongo transfer easier).
- macro_identifier contains:
	- the code for downloading a large collection of image macros
	- those images, as well as the products of applying dimensionality reduction algorithms (PCA and Randomized PCA) to them
	- the code for preprocessing these images and then building a classifier
	- graphs showing the accuracy of different classification/dimensionality reduction algorithms
- migration contains the translation file used with Mongify in order to transfer our SQL data to a Mongo database.
- text analytics contains code for reading jokes from our Mongo database, extracting features from those jokes, and building a classifier trained on these jokes that can categorize new jokes.
