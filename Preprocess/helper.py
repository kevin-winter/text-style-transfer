from bs4 import BeautifulSoup as BS
from collections import defaultdict
import numpy as np
import pickle as pkl
import logging
import sys
import re
import os

from nltk import word_tokenize, download, sent_tokenize
download('punkt')

import spacy
nlp = spacy.load('en')
nlp.max_length = 100000000

def configure_logging():
    logformat = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

    consoleLogger = logging.StreamHandler(sys.stdout)
    consoleLogger.setFormatter(logformat)
    consoleLogger.setLevel(logging.INFO)

    fileLogger = logging.FileHandler(filename='log.log', mode='a')
    fileLogger.setFormatter(logformat)
    fileLogger.setLevel(logging.DEBUG)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.addHandler(consoleLogger)
    logger.addHandler(fileLogger)


def clean_txt(txt):
    txt = re.sub("[^A-Za-z,.?! ]","", txt)
    txt = re.sub("(\r?\n)+"," ", txt)
    txt = re.sub(" +"," ", txt)
    return txt


def tokenize(txt):
    return [word_tokenize(sentence) for sentence in sent_tokenize(txt.lower())]


def crop_body(txt):
    txt = re.sub(r"\*\*\* ?start.*\*\*\*\r?\n", "***START***", txt, flags=re.I)
    txt = re.sub(r"\*\*\* ?end.*\*\*\*\r?\n", "***END***", txt, flags=re.I)
    txt = txt[txt.find("***START***")+11:txt.find("***END***")]
    return txt


def get_text(file):
    bs_file = BS(file, "lxml", from_encoding="UTF-8")
    return bs_file.find("text").getText()


def percentile_action(i, N, p, f, *fargs):
    if not i%np.ceil(N*p/100) or i==N:
        return f(i,N,p,fargs)
    else: 
        return False

def save_sents(i,N,p,args):
    assert len(args) == 2
    
    data = args[0]
    path = args[1] if args[1] else "."
    
    percent = i*100//N
    filepath = os.path.join(path,"tokens_{}.pkl".format(percent//p))
                            
    logging.info("{:3d}% parsed - saving to {}".format(percent, filepath))
    
    try:
        os.mkdir(path)
    except:
        pass
    
    with open(filepath, "wb") as out:
        pkl.dump(data, out)
    
    return True

def report(i,N,p,args):
    percent = i*100//N
    logging.info("{:3d}% parsed".format(percent))
    return True

def mask_named_entities(txt, filtered_types=["PERSON","ORG", "NORP","GPE"], maskwith=""):
    entities = defaultdict(set)
    doc = nlp(txt, disable=["tagger","parser"])

    for ent in doc.ents:
        if ent.label_ in filtered_types:
            entities[ent.label_].add(ent.text)

    for ent_type, ents in entities.items():
        for i,ent in enumerate(ents):
            replacement = maskwith if maskwith else "{}_{}".format(ent_type,i)
            txt = re.sub(" {} ".format(ent), " {} ".format(replacement), txt)
    
    return txt

def index(entity, iterable):
    try:
        return iterable.index(entity)
    except:
        return -1

def get_tags(txt):  
    doc = nlp(txt, disable=["parser", "ner"])
    return np.array(list(zip(*map(lambda x: (x.text, x.pos_, x.tag_), doc))))

def as_ints(feature):
    fdict = {v: i for i,v in enumerate(set(feature))}
    return list(map(lambda x: fdict[x], feature))

def w2v_sent_mapper(sents):
    return np.array(list(map(lambda sent: w2v_word_mapper(sent), sents)))

def w2v_word_mapper(words):
    return np.array(list(map(lambda word: get_word_vector(word), words)))
    
def get_word_vector(word):
    try:
        return w2v.wv[word]
    except:
        return w2v.wv["unknown"]