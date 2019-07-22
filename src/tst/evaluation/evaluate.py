import sys
import os
from os.path import join as pjoin

from keras.models import load_model

from tst.preprocess.markov import load_chain, load_pos_chain, load_emission_probs, beam_search, beam_search2
from tst.preprocess.transformers import TextCleaner
from tst.preprocess.translate import translate

sys.path.append('..')

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import word_tokenize

from tst.preprocess.helper import clean_txt
from tst.preprocess.w2v_extensions import *
from tst.preprocess.corpus import CorpusFileHandler
from tst.io import GB_DOCS, GB_DATA, AUTHORS, translation_embedding_dir


def sample_sentences(n, path=GB_DOCS):
    files = np.array(os.listdir(path))

    sents = []
    while True:
        fname = files[np.random.randint(0, len(files) - 1)]
        file = CorpusFileHandler(pjoin(GB_DOCS, fname))

        for x in file.as_token_stream():
            if 5 < len(x) < 25:
                if np.random.choice([0, 1], p=[.8, .2]):
                    sents.append(clean_txt(' '.join(x)))
                    if len(sents) >= n:
                        return sents


def p(sent):
    return word_tokenize(clean_txt(sent))


def bleu(original, translated, n=4):
    weights = [1 / n] * n
    return sentence_bleu([p(original)], p(translated), weights=weights, smoothing_function=SmoothingFunction().method7)


def cosine(x, y):
    return np.dot(np.divide(x, np.linalg.norm(x)), np.divide(y, np.linalg.norm(y)))


def sementic_similarity(original, translated, embedding):
    orig_vec = np.mean(embedding.vectorize(p(original)), axis=(0, 1))
    ref_vec = np.mean(embedding.vectorize(p(translated)), axis=(0, 1))
    return cosine(orig_vec, ref_vec)


def authorship_score(text, model, embedding):
    return predict(text, model, embedding)


def predict(sentence, model, embedding):
    text = TextCleaner().fit_transform(sentence)
    vecs = embedding.vectorize(text, flatten=True)
    vecs = pad(vecs, n=50)
    vecs = np.expand_dims(vecs, axis=0)
    return model.predict(vecs)[0][0]


def pad(arr, n):
    if arr.shape[0] >= n:
        return arr[:n]
    else:
        return np.concatenate([arr, np.zeros((n - arr.shape[0], arr.shape[1]))])


class EvaluationBench:
    def __init__(self, eval_embedding):
        self.emb = eval_embedding
        self.author = None
        self.chains, self.pos_chains, self.emission_probs, self.embeddings, self.models = {}, {}, {}, {}, {}

        t1 = lambda noise: lambda author: lambda sent: ' '.join(translate(sent,
                                                                          self.emb[author][0],
                                                                          self.emb[author][1],
                                                                          noise)[0])

        t2 = lambda author: lambda sent: beam_search(self.chains[author],
                                                     self.pos_chains[author],
                                                     self.emission_probs[author],
                                                     translate(sent,
                                                               self.embeddings[author][0],
                                                               self.embeddings[author][1], .2)[0],
                                                     beam_size=10,
                                                     word_trans_weight=.1,
                                                     emission_weight=.5,
                                                     context_weight=10,
                                                     eos_norm_weight=.2,
                                                     len_norm_weight=.15,
                                                     smoothing_prob=1e-5,
                                                     variable_length=True)[0]

        t3 = lambda author: lambda sent: beam_search2(self.chains[author],
                                                      self.pos_chains[author],
                                                      self.emission_probs[author],
                                                      translate(sent,
                                                                self.embeddings[author][0],
                                                                self.embeddings[author][1], .2)[0],
                                                      beam_size=10,
                                                      word_trans_weight=.5,
                                                      emission_weight=.5,
                                                      context_weight=1,
                                                      eos_norm_weight=.2,
                                                      len_norm_weight=.15,
                                                      smoothing_prob=1e-3,
                                                      variable_length=True)[0]

        self.methods = [t1(.1), t1(.2), t1(.3), t1(.4), t1(.5), t2, t3]
        self.selected_methods = []

    def _load_data(self, author):
        self.chains[author] = load_chain(author)
        self.pos_chains[author] = load_pos_chain(author)
        self.emission_probs[author] = load_emission_probs(author)

        folder = translation_embedding_dir(author)
        self.embeddings[author] = [KeyedVectors.load_word2vec_format(pjoin(folder, 'vectors-all.txt')),
                                   KeyedVectors.load_word2vec_format(pjoin(folder, f'vectors-{author}.txt'))]

        self.models[author] = load_model(pjoin(AUTHORS, author, 'parsed', 'clf.h5'))

    def set_author(self, author):
        self.author = author
        if author not in self.chains:
            self._load_data(author)

    def evaluate_sample(self, original, translated):
        bleu_score = bleu(original, translated)
        sim = sementic_similarity(original, translated, self.emb)
        auth_orig = authorship_score(original, self.models[self.author], self.emb)
        auth_trans = authorship_score(translated, self.models[self.author], self.emb)
        auth_diff = auth_trans - auth_orig

        return bleu_score, sim, auth_orig, auth_trans, auth_diff

    def evaluate(self):
        results = []
        with open(pjoin(GB_DATA, 'test_sentences.txt')) as f:
            sents = [l.strip() for l in f.readlines()]

        for i, sent in enumerate(sents):
            for j, translator in enumerate(self.selected_methods):
                for i_try in range(5):
                    try:
                        new_sent = translator(self.author)(sent)
                        break
                    except:
                        new_sent = ''
                scores = self.evaluate_sample(sent, new_sent)
                results.append([i, sent, j, new_sent, *scores])

        return results

