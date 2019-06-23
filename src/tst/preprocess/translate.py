from os.path import join as pjoin

import numpy as np
from gensim.models import KeyedVectors

from tst.io import translation_embedding_dir
from tst.preprocess.helper import tokenize
from tst.preprocess.w2v_extensions import *


def translate(sent, src_emb, target_emb, noise_level=.2):
    return np.array([[translate_word(w, src_emb, target_emb, noise_level) for w in s] for s in tokenize(sent)])


def translate_word(word, src_emb, target_emb, noise_level=.2):
    vec = src_emb.vector_or_zeros(word)
    if any(vec):
        return target_emb.similar_by_vector(vec + np.random.normal(0, noise_level, src_emb.vector_size), 1)[0][0]
    else:
        return word


def translate_to_author(sent, author, noise_level=.2):
    folder = translation_embedding_dir(author)
    default_emb = KeyedVectors.load_word2vec_format(pjoin(folder, 'vectors-all.txt'))
    author_emb = KeyedVectors.load_word2vec_format(pjoin(folder, f'vectors-{author}.txt'))
    return translate(sent, default_emb, author_emb, noise_level)