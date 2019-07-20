import os
from os.path import join as pjoin

import tst

BASE = os.path.abspath(pjoin(os.path.abspath(tst.__file__), '..', '..', '..'))
DATA = pjoin(BASE, 'data')
LOGS = pjoin(BASE, 'logs')
SRC = pjoin(BASE, 'src')
LIBS = pjoin(BASE, 'libs')

AUTHORS = pjoin(DATA, 'authors')
W2V = pjoin(AUTHORS, 'all', 'parsed', 'gutenberg_w2v_5e.model')

GB_DATA = pjoin(DATA, 'gbdata')
GB_CD = pjoin(GB_DATA, 'cd')
GB_DOCS = pjoin(GB_DATA, 'xml')

MUSE = pjoin(LIBS, 'muse')
GUTENTAG = pjoin(LIBS, 'GutenTag')
TRANSLATED = pjoin(MUSE, 'results')


def translation_embedding_dir(author):
    return sorted([pjoin(TRANSLATED, author, f) for f in os.listdir(pjoin(TRANSLATED, author))], key=os.path.getmtime)[-1]