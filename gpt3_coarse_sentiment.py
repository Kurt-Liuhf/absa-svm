import os
import sys
import re
import random
import configparser
from typing import Union, Tuple, List
from collections import defaultdict

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
nltk.download('wordnet')

def parse_filename(fp: str) -> Tuple[str, int, str]:
    _, review_type, sentiment, aspect = fp.split('_')
    return review_type, int(sentiment), aspect[:-4]

def load_generations(path: str, fname: str) -> Tuple[List[str], str, int, str]:
    review_type, sentiment, aspect = parse_filename(fname)
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    lines = [l.split('. ', 1)[1] for l in lines if len(l) > 2 and l[:1].isnumeric()]
    return lines, review_type, sentiment, aspect

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def get_sentiment(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
        #print('no tag')
        return []
    lemma = LEMMATIZER.lemmatize(word, pos=wn_tag)
    if not lemma:
        #print('no lemma')
        return []
    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        #print('no sysnet')
        return []
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())
    return [swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]

def coarse_aspect_score(sentence, aspect):
    tokens = nltk.word_tokenize(sentence)
    aspect_inds = [i for (i, t) in enumerate(tokens) if t.casefold() == aspect.casefold()]
    tags = nltk.pos_tag(tokens)
    senti_val = [get_sentiment(x,y) for (x,y) in tags]
    senti_val = [(tags[i][1], s) for (i, s) in enumerate(senti_val)]
    #print(tags)
    #print(senti_val)
    agg = []
    for i in aspect_inds:
        res = []
        for j, t in enumerate(senti_val):
            if i == j or not t[1] or t[0].startswith('NN'):
                continue
            elif t[0].startswith('J') or t[0].startswith('V'):
            #else:
                res.append((j, t))
        terms = [abs(i-r[0]) for r in res]
        if terms:
            #closest_terms = min(n)
            closest_terms = sorted(terms)[:CLOSEST_N]
            #print(closest_terms)
            for t in closest_terms:
                agg.append(res[terms.index(t)][1][1])
    return sum([a[0]-a[1] for a in agg])


def validate_aspect(data: List[str], length: int, aspect: str):
    # check if sentence actually contains the aspect, drop if does not
    data = [l for l in data if aspect in l.split(" ")]
    return data, len(data)

def coarse_aspect_score_efficiency(lines, target):
    total_sent = 0
    for l in lines:
        score = coarse_aspect_score(l, aspect)
        print(score, l)
        if target == 1 and score > 0:
            total_sent += 1
        elif target == -1 and score < 0:
            total_sent += 1
        elif target == 0 and score == 0:
            total_sent += 1
    return total_sent


if __name__ == '__main__':
    CLOSEST_N = 2
    LEMMATIZER = WordNetLemmatizer()
    PORTER_STEMMER = PorterStemmer()
    base_dir = 'datasets/rest/gens/'
    filename = 'raw_Restuarant_-1_service.txt'
    lines, review_type, sentiment, aspect = load_generations(base_dir+filename, filename)
    l1 = len(lines)
    lines, length = validate_aspect(lines, len(lines), aspect)
    l2 = len(lines)
    total_sent = coarse_aspect_score_efficiency(lines, sentiment)
    print(f'GPT3 Efficiency: {l2}/{l1}, {round(l2/l1, 2)}%')
    print(f'Coarse Aspect Score Efficiency: {total_sent}/{l1}, {round(total_sent/l1, 2)}%')

    #### WRITE RESULTS INTO ABSA.TXT FORMAT
    #### CONCAT FILES AFTER REVIEW