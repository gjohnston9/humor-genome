# adapted from: http://blog.yhat.com/posts/image-classification-in-Python.html

import numpy as np
import pandas as pd
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as pl
from PIL import Image

from IPython import embed

import os
import re


# set up a standard image size; this will distort some images but will get everything into the same shape
# TODO: allow for different file types? currently, different file types lead to different sizes
STANDARD_SIZE = (300, 250) # H x L
def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    if verbose == True:
        print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img


def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it 
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]


def get_reduced_dataset(num_components):
	"""
	reduce the dimensionality of the images dataset
	"""
	img_dir = "images/"
	images = [img_dir + f for f in os.listdir(img_dir)]
	labels = [re.split("[-/]", f)[-2] for f in images]

	data = []
	for image in images:
	    img = img_to_matrix(image, verbose = False)
	    img = flatten_image(img)
	    data.append(img)


	data = np.array(data)

	pca = RandomizedPCA(n_components = num_components)
	X = pca.fit_transform(data)

	columns = {chr(i+97) : X[:, i] for i in range(X.shape[1])}
	columns["label"] = labels

	df = pd.DataFrame(columns)

	return df


def plot_2d():
	"""
	reduce dimensionality of images dataset to 2, and plot the result
	"""

	data = get_reduced_dataset(2)

	colors = [
		"#FFC9D7",
		"#131313",
		"#FFB300",
		"#FFDB8B",
		"#803E75",
		"#FF6800",
		"#A6BDD7",
		"#C10020",
		"#CEA262",
		"#817066",
		"#007D34",
		"#F6768E",
		"#00538A",
		"#FF7A5C",
		"#53377A",
		"#FF8E00",
		"#B32851",
		"#F4C800",
		"#7F180D",
		"#93AA00",
		"#593315",
		"#F13A13",
		"#232C16",
	]

	for label, color in zip(data["label"].unique(), colors):
	    mask = data["label"] == label
	    pl.scatter(data[mask]["a"], data[mask]["b"], c=color, label=label, marker="s", s=[20] * len(data[mask]["a"]))
	pl.legend()
	pl.show()


if __name__ == "__main__":
	for i in range(1, 25):
		get_reduced_dataset(i).to_csv("reduced_images/images_{}_components.csv".format(i), index = False)
		print "finished getting reduced dataset with {} dimensions".format(i)
