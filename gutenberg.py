from bs4 import BeautifulSoup as BS
from zipfile import ZipFile

import pandas as pd
import numpy as np
import re


def parse_metadata(gutenberg_path):
    """Parses metadata files from Gutenberg DVD to create metadata dataframe."""
    try:
        metadf = pd.read_pickle("gutenberg_metadata.pkl")
        
    except:
        metadata = []

        metafiles = os.listdir(gutenberg_path+"/ETEXT")
        nr_files = len(metafiles)

        for i, metafile in enumerate(metafiles):
            book = {}

            with open(gutenberg_path+"/ETEXT/"+metafile, "r") as html:
                bs = BS(html)

            tables = bs.find_all("table")
            attrs = tables[0].find_all("tr")
            for attr in attrs:
                book[attr.find("th").text] = attr.find("td").text

            try:
                book["Path"] = bs.find("td", string=re.compile("text/plain")).findNext("a").get("href").replace("..","")
            except:
                book["Path"] = ""

            metadata.append(book)

            if not i%np.ceil(nr_files/10):
                print("{:3d}% parsed".format(i*100//nr_files))

        metadf = pd.DataFrame(metadata).fillna("")
        metadf["Release Date"] = pd.to_datetime(metadf["Release Date"]).fillna(0)
        pd.to_pickle(df,"gutenberg_metadata.pkl")
    
    return metadf


def filter_meta(metadf, author="", language="English",title="",years=""):
    """Applies conditions to existing metadata dataframe."""
    
    conditions = True
    
    if author:
        conditions &= metadf["Author"].str.contains(author)
    if language:
        conditions &= metadf["Language"].str.contains(language)
    if title:
        conditions &= metadf["Title"].str.contains(title)
    if years:
        dates = years.split("-")
        if len(dates) == 1:
            conditions &= metadf["Release Date"].isin(pd.date_range(dates[0], dates[0]+"-12-31"))
        elif len(dates) == 2:
            conditions &= metadf["Release Date"].isin(pd.date_range(dates[0] or "1700", dates[1]+"-12-31" or "2010-12-31"))
        
    return metadf[conditions]


def get_texts(metadf, path=""):
    """Obtain textfiles from all documents in the metadata dataframe. 
    If no DVD path is provided, the documents will be downloaded from www.gutenberg.org"""
    
    texts = []
    for i,book in metadf.iterrows():
        print("Loading EText {}: '{}' by {}".format(book["EText-No."], book["Title"], book["Author"]))
        try:  
            if path:
                with ZipFile(path + book["Path"]) as zfile:
                    txt = zfile.read(zfile.namelist()[0])
            else:
                txt = urlopen("http://www.gutenberg.org/files/{0}/{0}.txt".format(book["EText-No."])).read()

            texts.append(txt.decode("utf8","ignore"))
        except Exception as e:
            print("ERROR Could not load EText {}: {}".format(book["EText-No."], book["Title"]))
            print(e)
              
    return texts

def crop_body(txt):
    txt = re.sub(r"\*\*\* ?start.*\*\*\*\r?\n", "***START***", txt, flags=re.I)
    txt = re.sub(r"\*\*\* ?end.*\*\*\*\r?\n", "***END***", txt, flags=re.I)
    txt = txt[txt.find("***START***")+11:txt.find("***END***")]
    return txt

