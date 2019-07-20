import logging
import re
import os
from os.path import join as pjoin
from urllib.request import urlopen

from bs4 import BeautifulSoup as BS
from zipfile import ZipFile

import pandas as pd
import numpy as np
from shutil import copy

from tst.io import GB_CD, GB_DATA, GB_DOCS


def parse_metadata(gutenberg_path=GB_CD):
    """Parses metadata files from Gutenberg DVD to create metadata dataframe."""
    try:
        metadf = pd.read_pickle(pjoin(GB_DATA, "gutenberg_metadata.pkl"))

    except:
        metadata = []

        metafiles = os.listdir(pjoin(gutenberg_path, "/ETEXT"))
        nr_files = len(metafiles)

        for i, metafile in enumerate(metafiles):
            book = {}

            with open(pjoin(gutenberg_path, "ETEXT", metafile), "r") as html:
                bs = BS(html, "lxml")

            tables = bs.find_all("table")
            attrs = tables[0].find_all("tr")
            for attr in attrs:
                book[attr.find("th").text] = attr.find("td").text

            try:
                book["Path"] = bs.find("td", string=re.compile("text/plain"))\
                    .findNext("a").get("href").replace("..", "")
            except:
                book["Path"] = ""

            metadata.append(book)

            if not i % np.ceil(nr_files / 10):
                print("{:3d}% parsed".format(i * 100 // nr_files))

        metadf = pd.DataFrame(metadata).fillna("")
        metadf["Release Date"] = pd.to_datetime(metadf["Release Date"]).fillna(0)
        metadf.to_pickle(pjoin(GB_DATA, "gutenberg_metadata.pkl"))

    return metadf


def parse_xml_metadata(tei_folder_path=GB_DOCS):
    """Parses metadata files from Gutenberg TEI-XML files to create metadata dataframe."""
    try:
        metadata = pd.read_pickle(pjoin(GB_DATA, "gutenberg_metadata_fromxml.pkl"))
    except:
        docs = list(map(lambda x: pjoin(tei_folder_path, x), os.listdir(tei_folder_path)))

        meta = list(map(parse_xml_file, docs))
        metadata = pd.DataFrame(meta)

        pd.to_pickle(metadata, pjoin(GB_DATA, "gutenberg_metadata_fromxml.pkl"))

    return metadata


def parse_xml_file(filename):
    with open(filename, "r", errors="ignore") as xml:
        try:
            book = parse_xml_book(xml)
        except:
            book = {}

    book["path"] = filename
    return book


def parse_xml_book(xml):
    bs = BS(xml, "lxml")

    header = bs.tei.teiheader
    file = header.filedesc
    profile = header.profiledesc

    imprint = file.sourcedesc.biblstruct.monogr.imprint

    book = {}
    book["title"] = file.titlestmt.title.text
    book["lang"] = profile.langusage.language.text
    book["author_fullname"] = file.titlestmt.author.text

    if file.sourcedesc.listperson:
        author = file.sourcedesc.listperson.person
        book["author_forename"] = author.persname.forename.text
        book["author_lastname"] = author.persname.surname.text
        book["author_sex"] = author.sex.text
        book["author_nationality"] = author.nationality.text
        book["author_birthdate"] = author.birth.date.text
        book["author_deathdate"] = author.death.date.text

    book["year_published"] = imprint.date.text
    book["country_published"] = imprint.pubplace.country.text

    book["lc_class"] = profile.textclass.classcode.text
    book["keywords"] = ", ".join(map(lambda x: x.text, profile.findAll("term")))

    return book


def filter_meta(metadf, author="", language="English", title="", years=""):
    """Applies conditions to existing metadata dataframe."""

    conditions = True

    if author:
        conditions &= metadf["Author"].str.contains(author)
    if language:
        conditions &= metadf["Language"].str.contains(language)
    if title:
        conditions &= metadf["Title"].str.contains(title)
    if years:
        dates = years.split("-")
        if len(dates) == 1:
            conditions &= metadf["Release Date"].isin(pd.date_range(dates[0], dates[0] + "-12-31"))
        elif len(dates) == 2:
            conditions &= metadf["Release Date"].isin(
                pd.date_range(dates[0] or "1700", dates[1] + "-12-31" or "2010-12-31"))

    return metadf[conditions]


def get_texts(metadf, path=""):
    """Obtain textfiles from all documents in the metadata dataframe. 
    If no DVD path is provided, the documents will be downloaded from www.gutenberg.org"""

    texts = []
    for i, book in metadf.iterrows():
        logging.debug("Loading EText {}: '{}' by {}".format(book["EText-No."], book["Title"], book["Author"]))
        try:
            if path:
                with ZipFile(os.path.join(path, book["Path"])) as zfile:
                    txt = zfile.read(zfile.namelist()[0])
            else:
                txt = urlopen("http://www.gutenberg.org/files/{0}/{0}.txt".format(book["EText-No."])).read()

            texts.append(txt.decode("utf8", "ignore"))
        except Exception as e:
            logging.error("ERROR Could not load EText {}: {} ({})".format(book["EText-No."], book["Title"], e))

    return texts


def copy_files(df, src_folder, tgt_folder):
    os.makedirs(tgt_folder, exist_ok=True)
    for fname in df.path:
        copy(pjoin(src_folder, fname), tgt_folder)
