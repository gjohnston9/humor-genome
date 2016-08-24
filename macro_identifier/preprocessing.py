# adapted from: http://blog.yhat.com/posts/image-classification-in-Python.html

import numpy as np
import pandas as pd
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as pl
from PIL import Image

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

img_dir = "images/"
images = [img_dir + f for f in os.listdir(img_dir)]
labels = [re.split("[-/]", f)[-2] for f in images]

data = []
for image in images:
    img = img_to_matrix(image, verbose = False)
    img = flatten_image(img)
    data.append(img)


data = np.array(data)

pca = RandomizedPCA(n_components = 2) # TODO: after classifier is finished, see how changing number of components changes testing accuracy
X = pca.fit_transform(data)
df = pd.DataFrame({"x" : X[:, 0], "y" : X[:, 1], "label": labels})

# colors = [
# 	'#FF3333',  # red
# 	'#0198E1',  # blue
# 	'#BF5FFF',  # purple
# 	'#FCD116',  # yellow
# 	'#FF7216',  # orange
# 	'#4DBD33',  # green
# 	'#87421F'   # brown
# ]

df.to_pickle("images_2_components_df.p")

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

for label, color in zip(df["label"].unique(), colors):
    mask = df["label"] == label
    pl.scatter(df[mask]["x"], df[mask]["y"], c=color, label=label, marker="s", s=[20] * len(df[mask]["x"]))
pl.legend()
pl.show()

# clf = AdaBoostClassifier(n_estimators=5)
# scores = cross_val_score(clf, X, labels)
# print(scores.mean())