import os
import numpy as np
from bs4 import BeautifulSoup as BS
import pickle as pkl

from preprocess import tokenize


def get_text(file):
    bs_file = BS(file, from_encoding="UTF-8")
    return bs_file.find("text").getText()

def percentile_action(i, N, p, f, *fargs):
    if not i%np.ceil(N*p/100) or i==N:
        return f(i,N,p,fargs)
    else: 
        return False

def save_sents(i,N,p,args):
    percent = i*100//N
    print("{:3d}% parsed".format(percent))
    
    try:
        os.mkdir("tokens")
    except:
        pass
    
    with open("tokens/w2v_tokens_{}.pkl".format(percent//p), "wb") as out:
        pkl.dump(args[0], out)
    
    return True

def xml_to_tokens(path):
    files = os.listdir(filepath)
    nr_files = len(files)
    sents = []
    
    for i, fname in enumerate(files):
        with open(path + fname, "r", encoding="utf8") as file:
            try:
                sents += tokenize(get_text(file))
            except:
                print("Could not parse '{}'".format(fname))

        if percentile_action(i+1, nr_files, 5, save_sents, sents):
            sents = []