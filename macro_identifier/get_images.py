# adapted from: http://blog.yhat.com/posts/image-classification-in-Python.html
# list here: http://knowyourmeme.com/memes/advice-animals/children

from bs4 import BeautifulSoup
import requests
import re
import urllib2
import os

def get_soup(url):
    return BeautifulSoup(requests.get(url).text)

queries = ("good guy greg", "minor mistake marvin", "evil toddler",
	"unpopular opinion puffin", "religion pigeon", "bad luck brian",
	"sheltered college freshman", "bad joke eel", "confession bear",
	"first day on the internet kid", "sudden clarity clarence",
	"overly attached girlfriend", "college liberal", "lazy college senior",
	"advice god", "paranoid parrot", "high expectations asian father",
	"socially awkward penguin", "philosoraptor", "business cat")

for query in queries:
	url = "http://www.bing.com/images/search?q=" + query.replace(" ", "+") + "+meme+jpg"
	soup = get_soup(url)
	images = [a['src'] for a in soup.find_all("img", {"src": re.compile("mm.bing.net")})][:10]

	for num, img in enumerate(images):
	    raw_img = urllib2.urlopen(img).read()
	    f = open("images_test/" + query.replace(" ", "_") + "-"+ str(num), 'wb')
	    f.write(raw_img)
	    f.close()