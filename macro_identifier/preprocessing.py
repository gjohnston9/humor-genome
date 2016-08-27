# adapted from: http://blog.yhat.com/posts/image-classification-in-Python.html

import numpy as np
import pandas as pd
import sklearn.decomposition
import matplotlib.pyplot as pl

from sklearn.cross_validation import cross_val_score
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


def get_non_reduced_dataset(n = None):
    """
    transform the images for use with a classifier, but don't apply any dimensionality reduction
    """
    img_dir = "data/images/"
    images = [img_dir + f for f in os.listdir(img_dir)][:n]
    labels = [re.split("[-/]", f)[-2] for f in images][:n]

    data = []
    counter = 0
    for image in images:
        counter += 1
        img = img_to_matrix(image, verbose = False)
        img = flatten_image(img)
        data.append(img)
        if counter % 10 == 0:
            print "finished transforming {} of {} images".format(counter, len(images))


    data = np.array(data)

    columns = {i : column for i, column in enumerate(data.transpose())}
    columns["label"] = labels

    df = pd.DataFrame(columns)
    
    return df


def get_reduced_dataset(decomp_alg, *args, **kwargs):
    """
    reduce the dimensionality of the images dataset, using the supplied algorithm and parameters
    """
    img_dir = "data/images/"
    images = [img_dir + f for f in os.listdir(img_dir)]
    labels = [re.split("[-/]", f)[-2] for f in images]

    data = []
    for image in images:
        img = img_to_matrix(image, verbose = False)
        img = flatten_image(img)
        data.append(img)


    data = np.array(data)

    decomp = decomp_alg(*args, **kwargs)
    X = decomp.fit_transform(data)

    columns = {i : X[:, i] for i in range(X.shape[1])}
    columns["label"] = labels

    df = pd.DataFrame(columns)

    return df


def plot_2d():
    """
    reduce dimensionality of images dataset to 2, and plot the result
    """
    data = get_reduced_dataset(sklearn.decomposition.RandomizedPCA, n_components=2)

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
    # for i in range(1, 25):
    #     get_reduced_dataset(sklearn.decomposition.RandomizedPCA, n_components=i).to_csv("data/reduced_images/Randomized_PCA/{}_components.csv".format(i), index = False)
    #     print "finished getting reduced dataset with {} dimensions".format(i)

    for i in range(1,10):
        get_reduced_dataset(sklearn.decomposition.PCA, n_components=i).to_csv("data/reduced_images/PCA/{}_components.csv".format(i), index = False)
        print "finished getting reduced dataset with {} dimensions".format(i)    

    # get_non_reduced_dataset().to_csv("data/unreduced_images.csv", index = False)
