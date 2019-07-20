from itertools import islice

from nltk import word_tokenize, download, sent_tokenize
from bs4 import BeautifulSoup as BS
from gensim.models import Word2Vec
from multiprocessing import Pool, cpu_count
import numpy as np
import pickle as pkl
import logging
import spacy
import sys
import re
import os
from os.path import join as pjoin

from tst.io import W2V, LOGS


def w2v():
    global _w2v
    if "_w2v" not in globals():
        _w2v = Word2Vec.load(W2V)
    return _w2v


def nlp():
    global _nlp
    if "_nlp" not in globals():
        _nlp = spacy.load('en')
        _nlp.max_length = 1e8
    return _nlp


def init_config(logfile="_log.log", log_level=logging.DEBUG, w2v_path=None):
    global _w2v_path
    _w2v_path = w2v_path or W2V

    download('punkt')
    configure_logging(pjoin(LOGS, logfile), log_level)


def configure_logging(logfile, console=logging.INFO):
    logformat = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    consoleLogger = logging.StreamHandler(sys.stdout)
    consoleLogger.setFormatter(logformat)
    consoleLogger.setLevel(console)

    fileLogger = logging.FileHandler(filename=logfile, mode='a')
    fileLogger.setFormatter(logformat)
    fileLogger.setLevel(logging.DEBUG)

    logger.addHandler(consoleLogger)
    logger.addHandler(fileLogger)


def percentile_action(i, N, p, f, *fargs):
    if not i % np.ceil(N * p / 100) or i == N:
        return f(i, N, p, fargs)
    else:
        return False


def save_sents(i, N, p, args):
    assert len(args) == 2

    data = args[0]
    path = args[1] if args[1] else "."

    percent = i * 100 // N
    filepath = os.path.join(path, "tokens_{}.pkl".format(percent // p))

    logging.info("{:3d}% parsed - saving to {}".format(percent, filepath))

    os.makedirs(path, exist_ok=True)
    with open(filepath, "wb") as out:
        pkl.dump(data, out)

    return True


def report(i, N, p, args):
    percent = i * 100 // N
    logging.info("{:3d}% parsed".format(percent))
    return True


def index(entity, iterable):
    try:
        return iterable.index(entity)
    except:
        return -1


def clean_txt(txt):
    txt = re.sub("[^A-Za-z,.?! ]", "", txt)
    txt = re.sub("(\r?\n)+", " ", txt)
    txt = re.sub(" +", " ", txt)
    return txt


def tokenize(txt):
    if isinstance(txt, list):
        return [word_tokenize(sentence) for t in txt for sentence in sent_tokenize(t.lower())]
    else:
        return [word_tokenize(sentence) for sentence in sent_tokenize(txt.lower())]


def crop_body(txt):
    txt = re.sub(r"\*\*\* ?start.*\*\*\*\r?\n", "***START***", txt, flags=re.I)
    txt = re.sub(r"\*\*\* ?end.*\*\*\*\r?\n", "***END***", txt, flags=re.I)
    txt = txt[txt.find("***START***") + 11:txt.find("***END***")]
    return txt


def get_text(file):
    bs_file = BS(file, "lxml", from_encoding="UTF-8")
    text = bs_file.find("text")
    chapters = text.find_all('div', {'type': 'chapter'})
    if len(chapters) > 0:
        return [chap.getText() for chap in chapters if len(chap.getText()) > 10]
    else:
        return [text.getText()]


def psum(it):
    pool = Pool(cpu_count())
    for i in range(int(np.ceil(np.log2(len(it))))):
        it = pool.imap(adder, chunk(it, 2))
    return list(it)[0]


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def adder(x):
    l = list(x)
    return l[0] if len(x) == 1 else l[0] + l[1] if len(l) == 2 else sum(l)
