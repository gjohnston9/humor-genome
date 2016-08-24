# adapted from: http://blog.yhat.com/posts/image-classification-in-Python.html

import numpy as np
import pandas as pd
from sklearn.decomposition import RandomizedPCA
import matplotlib.pyplot as pl

from PIL import Image
import os

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

img_dir = "images/"
images = [img_dir + f for f in os.listdir(img_dir)]
labels = [f.split("-")[-2] for f in images]

data = []
for image in images:
    img = img_to_matrix(image, verbose = False)
    img = flatten_image(img)
    data.append(img)


data = np.array(data)

pca = RandomizedPCA(n_components = 2) # TODO: after classifier is finished, see how changing number of components changes testing accuracy
X = pca.fit_transform(data)
df = pd.DataFrame({"x" : X[:, 0], "y" : X[:, 1], "label": labels})

colors = [
	'#FF3333',  # red
	'#0198E1',  # blue
	'#BF5FFF',  # purple
	'#FCD116',  # yellow
	'#FF7216',  # orange
	'#4DBD33',  # green
	'#87421F'   # brown
]

for label, color in zip(df["label"].unique(), colors):
    mask = df["label"] == label
    pl.scatter(df[mask]["x"], df[mask]["y"], c=color, label=label, marker="s")
pl.legend()
pl.show()
