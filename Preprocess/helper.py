from nltk import word_tokenize, download, sent_tokenize
from bs4 import BeautifulSoup as BS
from collections import defaultdict
from gensim.models import Word2Vec
import numpy as np
import pickle as pkl
import logging
import spacy
import sys
import re
import os

def w2v():
    global _w2v
    if "_w2v" not in globals():
        _w2v = Word2Vec.load(_w2v_path)
    return _w2v

def nlp():
    global _nlp
    if "_nlp" not in globals():
        _nlp = spacy.load('en')
        _nlp.max_length = 100000000
    return _nlp

def init_config(logfile= "_log.log", w2v_path = None):
    global _w2v_path
    filepath = os.path.abspath(os.path.dirname(__file__))
    _w2v_path = w2v_path if w2v_path else os.path.join(filepath,"w2v_models/gutenberg_w2v_5e.model")
    
    download('punkt')
    configure_logging(logfile)
    
def configure_logging(logfile):
    logformat = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

    consoleLogger = logging.StreamHandler(sys.stdout)
    consoleLogger.setFormatter(logformat)
    consoleLogger.setLevel(logging.INFO)

    fileLogger = logging.FileHandler(filename=logfile, mode='a')
    fileLogger.setFormatter(logformat)
    fileLogger.setLevel(logging.DEBUG)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.addHandler(consoleLogger)
    logger.addHandler(fileLogger)

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

def index(entity, iterable):
    try:
        return iterable.index(entity)
    except:
        return -1




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

def w2v_sent_mapper(sents):
    return np.array(list(map(lambda sent: w2v_word_mapper(sent), sents)))

def w2v_word_mapper(words):
    return np.array(list(map(lambda word: get_word_vector(word), words)))
    
def get_word_vector(word):
    try:
        return w2v().wv[word]
    except:
        return w2v().wv["unknown"]