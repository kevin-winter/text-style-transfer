import os
import re
import pickle as pkl

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.image import extract_patches
from sklearn.model_selection import train_test_split

from collections import defaultdict
import pandas as pd
from scipy.sparse.csr import csr_matrix

# SELECTORS
from tst.preprocess.helper import nlp


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns, byname=False, inverse=False):
        self.byname = byname
        self.columns = columns
        self.inverse = inverse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols = self.columns if self.byname else X.columns[self.columns]
        mask = X.columns.isin(cols)
        return X.loc[:, ~mask] if self.inverse else X.loc[:, mask]


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtypes):
        self.dtypes = dtypes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.select_dtypes(include=self.dtypes)

    # TRANSFORMERS


class NamedEntityMasker(BaseEstimator, TransformerMixin):
    def __init__(self, filtered_types=["PERSON", "ORG", "NORP", "GPE"], maskwith=""):
        self.filtered_types = filtered_types
        self.maskwith = maskwith

    def fit(self, X, y=None):
        self.entities = defaultdict(set)
        return self

    def transform(self, X):
        doc = nlp()(X, disable=["tagger", "parser"])

        for ent in doc.ents:
            if ent.label_ in self.filtered_types:
                self.entities[ent.label_].add(ent.text)

        for ent_type, ents in self.entities.items():
            for i, ent in enumerate(ents):
                replacement = self.maskwith if self.maskwith else "{}_{}".format(ent_type, i)
                X = re.sub(" {} ".format(ent), " {} ".format(replacement), X)

        return X


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_mapping=lambda x: (x.text, x.i - x.sent.start, x.sent.end - x.i, x.pos_, x.tag_)):
        self.feature_mapping = feature_mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        doc = nlp()(X, disable=["ner"])
        return pd.DataFrame(list(map(self.feature_mapping, doc)))


class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, punct_spacing=True):
        self.punct_spacing = punct_spacing

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = re.sub("([\r]?\n)+", " ", X)
        X = re.sub("[^A-Za-z,.?! ]", "", X)
        if not self.punct_spacing:
            X = re.sub(" ([.!?,])", "\g<1>", X)
        else:
            X = re.sub("([.!?,])", " \g<1>", X)
        X = re.sub("[ ]+", " ", X)
        return X


class TextVectorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return w2v_word_mapper(X.iloc[:, 0])


class SpacyAnalyser(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return nlp()(X)


class PatchExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, patchsize, stepsize):
        self.patchsize = patchsize
        self.stepsize = stepsize

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if type(X) == csr_matrix:
            X = X.toarray()
        if type(X) == pd.DataFrame:
            X = X.as_matrix()
        return np.rollaxis(extract_patches(X, (self.patchsize, X.shape[1]), self.stepsize), 1, 4)


class FeatureEncoder(Pipeline):
    def __init__(self, patchsize, patchstep):
        super().__init__([("_union", FeatureUnion([
            ("_W2V", Pipeline([
                ("_ColumnSelector_Words", ColumnSelector([0])),
                ("TextVectorizer", TextVectorizer())
            ])),
            ("_OtherFeatures", Pipeline([
                ("_ColumnSelector_!Words", ColumnSelector([0], inverse=True)),
                ("_union", FeatureUnion([
                    ("_discrete_features", Pipeline([
                        ("TypeSelector_Discrete", TypeSelector(["object", "category"])),
                        ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore"))
                    ])),
                    ("_continous_features", Pipeline([
                        ("TypeSelector_Continuous", TypeSelector(["number"])),
                        ("MinMaxScaler", MinMaxScaler(feature_range=[-1, 1]))
                    ]))
                ]))
            ])),
        ])),
                          ("PatchExtractor", PatchExtractor(patchsize, patchstep))
                          ])


class TextParser(Pipeline):
    def __init__(self):
        super().__init__([
            ("TextCleaner", TextCleaner()),
            ("NamedEntityMasker", NamedEntityMasker(["PERSON"], maskwith="person")),
            ("TextFeatureExtractor", TextFeatureExtractor())
        ])


# HELPER


def apply_pipeline(X, pipeline, chunksize):
    Xt = None
    start = 0
    chunksize = 10 ** 6
    while True:
        print("Iteration {} / {}".format(start // chunksize, len(X) // chunksize), end="\r")
        part = X[start:start + chunksize]
        transformed = pipeline.fit_transform(part)

        if type(transformed) == csr_matrix:
            transformed = pd.DataFrame(transformed.todense())
        elif type(transformed) in (np.ndarray, np.array, pd.DataFrame):
            transformed = pd.DataFrame(transformed)

        Xt = pd.concat((Xt, transformed), ignore_index=True)

        if len(part) < chunksize:
            break

        start += chunksize

    return Xt


def get_data(authors):
    features = list(map(load_features, authors))

    labels = np.concatenate([np.ones(l) * i for i, l in enumerate(map(len, features))])
    features = pd.concat(features, ignore_index=True)

    patchsize, patchstep = 100, 50
    X = FeatureEncoder(patchsize, patchstep).fit_transform(features)
    y = np.median(extract_patches(labels, patchsize, patchstep), axis=1)

    return train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)


def load_features(author):
    path = os.path.join(os.path.dirname(__file__), "../data/data/{0}/{0}_features.pkl".format(author))
    with open(path, "rb") as f:
        features = pkl.load(f)
    return features


def load_text(author):
    path = os.path.join(os.path.dirname(__file__), "../data/data/{0}/{0}_string.pkl".format(author))
    with open(path, "rb") as f:
        string = pkl.load(f)
    return string
