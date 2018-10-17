import numpy as np
from bs4 import BeautifulSoup as BS
import pickle as pkl
from nltk import word_tokenize, download, sent_tokenize
download('punkt')


def clean_txt(txt):
    txt = crop_body(txt)
    #txt = txt.lower()
    txt = re.sub("(\r?\n)+"," ", txt)
    txt = re.sub(" +"," ", txt)
    txt = "".join(list(filter(lambda x: x not in '"#$%&\'()*+-/:;<=>@[\\]^_`{|}~', txt)))
    return txt

def tokenize(txt):
    return [word_tokenize(sentence) for sentence in sent_tokenize(txt.lower())]

def crop_body(txt):
    txt = re.sub(r"\*\*\* ?start.*\*\*\*\r?\n", "***START***", txt, flags=re.I)
    txt = re.sub(r"\*\*\* ?end.*\*\*\*\r?\n", "***END***", txt, flags=re.I)
    txt = txt[txt.find("***START***")+11:txt.find("***END***")]
    return txt

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
    logging.info("{:3d}% parsed".format(percent))
    
    try:
        os.mkdir("tokens")
    except:
        pass
    
    with open("tokens/w2v_tokens_{}.pkl".format(percent//p), "wb") as out:
        pkl.dump(args[0], out)
    
    return True

def report(i,N,p,args):
    percent = i*100//N
    logging.info("{:3d}% parsed".format(percent))
    return True