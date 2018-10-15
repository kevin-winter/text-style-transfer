from nltk import word_tokenize, download, sent_tokenize
download('punkt')


def clean_txt(txt):
    print("Preprocess: Clean text")
    txt = crop_body(txt)
    #txt = txt.lower()
    txt = re.sub("(\r?\n)+"," ", txt)
    txt = re.sub(" +"," ", txt)
    txt = "".join(list(filter(lambda x: x not in '"#$%&\'()*+-/:;<=>@[\\]^_`{|}~', txt)))
    return txt

def tokenize(txt):
    print("Preprocess: Tokenize")
    return [word_tokenize(sentence) for sentence in sent_tokenize(cleaned)]