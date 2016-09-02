import pandas as pd
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

from IPython import embed


def cross_validate_knn_unreduced(k_range, verbose = False):
	# df = pd.read_csv("data/unreduced_images.csv")

	tp = pd.read_csv("data/unreduced_images.csv", iterator = True, chunksize = 5)
	
	if verbose:
		print "finished reading CSV. concatenating chunks that were read."
	
	df = pd.concat(tp, ignore_index = True)
	
	if verbose:
		print "finished concatenating."

	all_scores = []

	for k in k_range:
		if verbose:
			print "using k={} for KNN".format(k)

		samples = zip(*(df[str(j)] for j in range(sum(1 for k in df.keys() if k.isdigit()))))
		targets = df["label"]

		# embed()

		scores = cross_val_score(KNeighborsClassifier(n_neighbors = k), samples, targets, cv = 5)
		avg = sum(scores) / len(scores)

		if verbose:
			print avg

		all_scores.append(avg)

	pl.plot(k_range, all_scores, marker = "o")

	pl.xlabel("number of neighbors used in KNN")
	pl.ylabel("average accuracy during cross validation")
	pl.title("numbers of neighbors used in KNN vs. cross validation accuracy")
	pl.grid()

	pl.show()


def cross_validate_knn_reduced(k_range, n_components_range, path, verbose = False):
	decomp_name = path.split("/")[-1]
	for k in k_range:
		all_scores = []
		for i in n_components_range:
			if verbose:
				print "using {} components".format(i)

			df = pd.read_csv(path + "/{}_components.csv".format(i))

			samples = zip(*(df[str(j)] for j in range(i)))
			targets = df["label"]

			# x_train, x_test, y_train, y_test = train_test_split(samples, targets, test_size = 0.33)
			# classifier = KNeighborsClassifier()
			# classifier.fit(x_train, y_train)
			# predicted = classifier.predict(x_test)
			# print "Classification report for classifier:\n{}".format(metrics.classification_report(y_test, predicted))
			# print "Confusion matrix:\n{}".format(metrics.confusion_matrix(y_test, predicted))
			# print "accuracy: {}".format(metrics.accuracy_score(y_test, predicted))

			scores = cross_val_score(KNeighborsClassifier(n_neighbors = k), samples, targets, cv = 5)
			avg = sum(scores) / len (scores)

			if verbose:
				print avg

			all_scores.append(avg)

		pl.plot(n_components_range, all_scores, marker = "o", label = "KNN, k={}".format(k))
		if verbose:
			print "\n\n\n"

	pl.xlabel("number of components")
	pl.ylabel("average accuracy during cross validation")
	pl.title("\nnumber of components \n (reduced from original dataset using {}) \n vs. cross validation accuracy".format(decomp_name))
	pl.legend()
	pl.grid()

	with PdfPages("results/KNN-{}.pdf".format(decomp_name)) as pdf:
		pdf.savefig()

	# pl.show()

	pl.clf()

if __name__ == "__main__":
	cross_validate_knn_reduced(range(1,7), range(1,25), "data/reduced_images/Randomized_PCA", verbose = False)
	cross_validate_knn_reduced(range(1,7), range(1,25), "data/reduced_images/PCA", verbose = False)
	# cross_validate_knn_unreduced(range(1,7), verbose = True) # don't try to do this on laptop...