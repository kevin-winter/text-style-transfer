import logging
import os
from os.path import join as pjoin
import pickle as pkl
import re
from collections import defaultdict, Counter

import markovify
import numpy as np
import pandas as pd
import spacy
import wordfreq
from networkx import from_dict_of_dicts, shortest_path
from sklearn.pipeline import Pipeline

from tst.io import AUTHORS
from tst.preprocess.corpus_helper import CorpusStreamer
from tst.preprocess.helper import configure_logging, nlp, psum
from tst.preprocess.transformers import TextFeatureExtractor, TextCleaner


def starts_with_vowel(word):
    return word.lower()[0] in ["a", "e", "i", "o", "u"]


def lexical_freq(word):
    return wordfreq.zipf_frequency(word, lang="en")


def bining(number, nbins):
    return min(int(number / (1 / nbins)), nbins - 1)


def mapping(word):
    return (word.tag_,
            word.is_stop * 1,
            bining(len(word) / 15, 3),
            bining(lexical_freq(word.text) / 10, 3),
            word.dep_,
            bining(word.i / word.sent.end, 3))


class TextParser(Pipeline):
    def __init__(self):
        super().__init__([
            ("TextCleaner", TextCleaner()),
            ("TextFeatureExtractor", TextFeatureExtractor(mapping))
        ])


def to_style_tokens(text):
    features = TextParser().fit_transform(text)
    tokens_text = " ".join(features.apply(lambda x: "_".join(map(str, x)), axis=1))
    return re.sub(" ([.!?])[^ ]+", "\g<1>", tokens_text)


def limit_style_tokens(word, length=4):
    return '_'.join(word.split('_')[:length])


def find_token_limiting(nodes, word):
    np.random.shuffle(nodes)
    short_nodes = list(map(limit_style_tokens, nodes))
    return nodes[short_nodes.index(limit_style_tokens(word))]


def pos_emission_prob(author):
    docs_path = pjoin(AUTHORS, author, 'books')
    corpus = CorpusStreamer(docs_path, False)
    counts = defaultdict(Counter)

    for text in corpus:
        doc = nlp()(TextCleaner().fit_transform(text))
        for w, t in zip(doc, map(lambda x: '_'.join(map(str, x)), map(mapping, doc))):
            counts[t][w.orth_.lower()] += 1

    save_folder = pjoin(AUTHORS, author, 'parsed')
    os.makedirs(save_folder, exist_ok=True)

    with open(pjoin(save_folder, 'emission_probs.pkl'), 'wb') as f:
        pkl.dump(counts, f)

    return counts


def load_emission_probs(author):
    with open(pjoin(AUTHORS, author, 'parsed', 'emission_probs.pkl'), 'rb') as f:
        return pkl.load(f)


class LowerMarkovifyText(markovify.Text):
    def word_split(self, sentence):
        return re.split(super().word_split_pattern, sentence.lower())


def pos_markov_chain(author, state_size=2):
    docs_path = pjoin(AUTHORS, author, 'books')
    corpus = CorpusStreamer(docs_path, False)
    saved_chain = None
    saved_pos_chain = None

    for i, text in enumerate(corpus):
        logging.debug('POS_MARKOV_CHAIN: {: >5} texts parsed'.format(i + 1))
        chain = LowerMarkovifyText(TextCleaner(False).transform(text), state_size=state_size, retain_original=False)
        saved_chain = markovify.combine([saved_chain, chain]) if saved_chain else chain

        pos_chain = markovify.Text(to_style_tokens(text), state_size=state_size, retain_original=False)
        saved_pos_chain = markovify.combine([saved_pos_chain, pos_chain]) if saved_pos_chain else pos_chain

    save_folder = pjoin(AUTHORS, author, 'parsed')
    os.makedirs(save_folder, exist_ok=True)

    with open(os.path.join(save_folder, 'word_mm.json'), 'w') as f:
        f.write(saved_chain.to_json())

    with open(os.path.join(save_folder, 'pos_mm.json'), 'w') as f:
        f.write(saved_pos_chain.to_json())

    return saved_chain, saved_pos_chain


def load_chain(author):
    with open(pjoin(AUTHORS, author, 'parsed', 'word_mm.json'), 'r') as f:
        return LowerMarkovifyText.from_json(f.read())


def load_pos_chain(author):
    with open(pjoin(AUTHORS, author, 'parsed', 'pos_mm.json'), 'r') as f:
        return markovify.Text.from_json(f.read())


def vocabulary(chain):
    cnts = list(Counter(x) for x in chain.chain.model.values())
    vocab = psum(cnts)
    return vocab


def markov_to_graph(markovchain):
    markovmodel = {}

    for k, v in markovchain.chain.model.items():
        _sum = sum(v.values())

        markovmodel[k[-1]] = {_k: {"weight": -np.log(_v / _sum)} for _k, _v in v.items()}

    return from_dict_of_dicts(markovmodel)


def make_sentence_containing(markovchain, words, tokenize=True, strict=False):
    g = markov_to_graph(markovchain)

    if tokenize:
        words = [find_token_limiting(list(g.nodes), to_style_tokens(word)) for word in words]

    if not strict:
        np.random.shuffle(words)

    tokens = ['___BEGIN__'] + words + ['___END__']
    sentence = []
    for i in range(len(tokens) - 1):
        path = shortest_path(g, tokens[i], tokens[i + 1], weight="weight")[1:]
        sentence.extend(path)

    return ' '.join(sentence[:-1])


def find_sentence_containing(markovchain, words, tokenize=True, max_tries=10000, max_words=25):
    if tokenize:
        words = list(map(limit_style_tokens, map(to_style_tokens, words)))

    for i in range(max_tries):
        sent = markovchain.make_sentence(tries=100, max_words=max_words)
        tmp_sent = sent

        for word in words:
            if word in tmp_sent:
                tmp_sent = tmp_sent.replace(word, '', 1)
            else:
                sent = ''
                break

        if sent:
            print('Finding a sentence took {} tries.'.format(i))
            return sent
        else:
            continue


def safe_find(array, item):
    try:
        return array.index(item)
    except:
        return -1


def log_normalize_dict(d):
    if isinstance(d, list):
        d = {k: 1 for k in d}

    norm_sum = sum(d.values())
    return {k: np.log(v / norm_sum) for k, v in d.items()}


def normalize_dict(d):
    if isinstance(d, list):
        d = {k: 1 for k in d}

    norm_sum = sum(d.values())
    return {k: v / norm_sum for k, v in d.items()}


def len_norm(n, t, weight=.7):
    if n == 0:
        return 1

    norm = n / (1 / n - weight * (n - t) ** 2 / (1 - t) ** 2)
    #     norm = (5 + n) ** weight / (5 + 1) ** weight
    #     norm = n ** weight if n > 5 else n
    return norm


def eos_norm(n, t, weight=0.2, log=True):
    norm = (n + 1) / t
    return weight * np.log(norm) if log else norm


def beam_search(word_mm, pos_mm, emission_probs, context_words, beam_size=5, smoothing_prob=1e-6,
                word_trans_weight=.5, emission_weight=.5, context_weight=.05, eos_norm_weight=.2, len_norm_weight=.7,
                begin_token='___BEGIN__', end_token='___END__',
                variable_length=True, max_length=30):
    weights = [word_trans_weight, emission_weight, context_weight]
    word_trans_weight, emission_weight, context_weight = np.array(weights) / sum(weights)

    pos_sent = find_sentence_containing(pos_mm, context_words, max_words=max_length).split()
    n_t = len(pos_sent)
    print(n_t)
    #     pos_sent = pos_mm.make_sentence(max_words=max_length).split()

    queue = {tuple(): 0}
    i = 0

    while True:
        cur_tag = pos_sent[i] if i < len(pos_sent) else ''
        layer_candidates = queue if variable_length and i != 0 else {}

        for prev_words, prev_score in queue.items():
            # cur_tag = pos_sent[len(prev_words)] if len(prev_words) < len(pos_sent) else ''
            if (prev_words and prev_words[-1] == end_token) or len(prev_words) == max_length:
                if not variable_length:
                    layer_candidates = {**layer_candidates, **{prev_words: prev_score}}
                continue

            n = len(prev_words)
            prev_state = [begin_token] * (word_mm.chain.state_size - n) + list(prev_words[-word_mm.chain.state_size:])

            transition_candidates = normalize_dict(word_mm.chain.model.get(tuple(prev_state), {}))
            emission_candidates = normalize_dict(emission_probs[cur_tag])
            context_candidates = normalize_dict(context_words)

            merged_probs = pd.DataFrame([transition_candidates, emission_candidates, context_candidates]) \
                .fillna(smoothing_prob) \
                .apply(lambda x: sum(x * [word_trans_weight, emission_weight, context_weight]))

            merged_probs[end_token] = merged_probs.get(end_token, smoothing_prob) + eos_norm(n, n_t, eos_norm_weight,
                                                                                             False)

            reduction_words = set(prev_words[-3:]) or (set(prev_words) and set(context_words))
            for word in reduction_words:
                merged_probs[word] = smoothing_prob if word in merged_probs else 0

            merged_probs /= sum(merged_probs)
            merged_log_probs = np.log(merged_probs)

            selected_candidates = merged_log_probs.nlargest(beam_size)
            selected_candidates = {
                tuple(prev_words) + (word,): (prev_score * len_norm(n, n_t, len_norm_weight) + score) / len_norm(n + 1,
                                                                                                                 n_t,
                                                                                                                 len_norm_weight)
                for word, score in selected_candidates.items()}
            layer_candidates = {**layer_candidates, **selected_candidates}

        old_queue = queue
        queue = {k: layer_candidates[k] for k in
                 sorted(layer_candidates, key=layer_candidates.get, reverse=True)[:beam_size]}

        if set(queue.keys()) == set(old_queue.keys()):
            break

        #         for words, score in queue.items():
        #             print(' '.join(words), score)
        #         print()
        i += 1

    results = dict(filter(lambda x: end_token in x[0], queue.items()))
    if len(results) == 0:
        results = queue
        best = max(results, key=results.get)
        sent = ' '.join(best)
    else:
        best = max(results, key=results.get)
        sent = ' '.join(best[:safe_find(best, end_token)])

    print(sent)
    return sent, queue[best]
