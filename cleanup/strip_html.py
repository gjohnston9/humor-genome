# setup: sudo -h pip2 install unicodecsv
import unicodecsv as csv
import pdb
from HTMLParser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

with open("all_dirty.csv", "rb") as input_csv, open("all_cleaned.csv", "wb") as output_csv:
    csv_reader = csv.reader(input_csv, delimiter="\t", encoding="utf-8")
    csv_writer = csv.writer(output_csv, delimiter="\t", encoding="utf-8")
    h = HTMLParser()
    first = True
    for row_num, row in enumerate(csv_reader):
        if first:
            csv_writer.writerow(row)
            first = False
        else:
            unescaped = h.unescape(row[1])
            stripped = strip_tags(unescaped)
            csv_writer.writerow([row[0], stripped, row[2], row[3], row[4], row[5]])
