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
