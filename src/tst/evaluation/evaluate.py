import sys
import os
from os.path import join as pjoin
sys.path.append('..')

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
import numpy as np

from tst.preprocess.helper import tokenize, clean_txt
from tst.preprocess.w2v_extensions import *
from tst.preprocess.corpus import CorpusFileHandler
from tst.io import W2V, GB_DOCS, GB_DATA


def sample_sentences(n, path=GB_DOCS):
    files = np.array(os.listdir(path))
    selected_files = files[np.random.randint(0, len(files)-1, n)]
    
    sents = []
    while True:
        fname = files[np.random.randint(0, len(files)-1)]
        file = CorpusFileHandler(pjoin(GB_DOCS, fname))
        
        for x in file.as_token_stream():
            if 5 < len(x) < 25:
                if np.random.choice([0,1], p=[.8,.2]):
                    sents.append(clean_txt(' '.join(x)))
                    if len(sents) >= n:
                        return sents
        
    return sents

def p(sent):
    return word_tokenize(clean_txt(sent))

def bleu(original, translated, n=4):
    weights = [1/n]*n
    return sentence_bleu([p(original)], p(translated), weights=weights, smoothing_function=SmoothingFunction().method7)

def cosine(x, y):
    return np.dot(np.divide(x, np.linalg.norm(x)), np.divide(y, np.linalg.norm(y)))

def sementic_similarity(original, translated, embedding):
    orig_vec = np.mean(embedding.vectorize(p(original)), axis=(0,1))
    ref_vec = np.mean(embedding.vectorize(p(translated)), axis=(0,1))
    return cosine(orig_vec, ref_vec)
    
def authorship_score(text, author):
    return 1

def evaluate_sample(original, translated, embedding, author):
    bleu_score = bleu(original, translated)
    sim = sementic_similarity(original, translated, embedding)
    auth_orig = authorship_score(original, author)
    auth_trans= authorship_score(translated, author)
    auth_diff = auth_trans - auth_orig
    
    return bleu_score, sim, auth_orig, auth_trans, auth_diff

def evaluate(translators, author, embedding):
    results = []
    with open(pjoin(GB_DATA, 'test_sentences.txt')) as f:
        sents = [l.strip() for l in f.readlines()]
    
    for i, sent in enumerate(sents):
        for j, translator in enumerate(translators):
            new_sent = translator(sent)
            scores = evaluate_sample(sent, new_sent, embedding, author)
            results.append([i, sent, j, new_sent, *scores])
    
    return results
