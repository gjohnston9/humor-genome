import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":
	verbose = False
	for k in range(1,7):
		all_scores = []
		for i in range(2,25):
			if verbose:
				print "using {} components".format(i)
			df = pd.read_csv("reduced_images/images_{}_components.csv".format(i))
			samples = zip(*(df[chr(j+97)] for j in range(i)))
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

		pl.plot(range(2,25), all_scores, marker = "o", label = "KNN, k={}".format(k))
		if verbose:
			print "\n\n\n"

	pl.xlabel("number of components")
	pl.ylabel("average accuracy during cross validation")
	pl.title("number of components vs. cross validation accuracy")
	pl.legend()
	pl.grid()
	# pl.show()

	with PdfPages("knn_randomizedPCA.pdf") as pdf:
		pdf.savefig()