import os
import re
import numpy as np

module_path = os.path.dirname(__file__)

LRB_RE = re.compile(r'-LRB-')
RRB_RE = re.compile(r'-RRB-')
LSB_RE = re.compile(r'-LSB-')
RSB_RE = re.compile(r'-RSB-')
WORD_RE = re.compile(r'\b.?w*.\b')

negation = ["never", "no", "nothing", "nowhere", "noone", "none", "not", \
            "havent", "hasnt", "hadnt", "cant", "couldnt", "shouldnt", "wont", "wouldnt", "dont", "doesnt", "didnt",
            "isnt", "arent", "aint", \
            "haven't", "hasn't", "hadn't", "can't", "couldn't", "shouldn't", "won't", "wouldn't", "don't", "doesn't",
            "didn't", "isn't", "aren't", "ain't"]
posStop = [",", ".", ":"]
posAdd = ["NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS", "RB", "RBR", "RBS"]


class Lexicons:

    def __init__(self, lex_name, words, pos_tags, uni_path=None, bi_path=None, pos_path=None, neg_path=None):
        self.lex_name = lex_name
        self.words = words
        self.pos_tags = pos_tags
        self.scores = []
        # load lexicons
        if uni_path is not None:
            self.uni_lex = Lexicons.load_lexicons(uni_path)
        if bi_path is not None:
            self.bi_lex = Lexicons.load_lexicons(bi_path)
        if pos_path is not None and neg_path is not None:
            self.uni_lex = Lexicons.load_pos_neg_lex(pos_path, neg_path)
        # calculate scores
        if bi_path is None:
            self.scores = self.get_score(flag=1)
        else:
            self.scores = self.get_score(flag=2)

    @staticmethod
    def load_lexicons(path):
        lex = {}
        with open(module_path + path, encoding='utf-8') as f:
            for line in f.readlines():
                tmp = line.strip().split('\t')
                lex[tmp[0]] = float(tmp[1])
        return lex

    @staticmethod
    def load_pos_neg_lex(pos_path, neg_path):
        lex = {}
        pos_lex = [l.strip().lower() for l in open(module_path + pos_path).readlines()]
        neg_lex = [l.strip().lower() for l in open(module_path + neg_path).readlines()]
        for w in pos_lex:
            lex[w] = 1.0
        for w in neg_lex:
            lex[w] = -1.0

        return lex

    @staticmethod
    def parse_lex(words, pos_tags, flag=0):
        if flag == 0:
            for w in words:
                w = w.lower()
        elif flag == 1:    # for unigram extraction
            negflag = 0
            negfirst = '_NEGFIRST'
            neg = '_NEG'

            for w, pos in zip(words, pos_tags):
                if pos in posStop:
                    negflag = 0

                if pos in posAdd:
                    if negflag == 1:
                        negflag += 1
                        w += negfirst
                    elif negflag > 1:
                        w += neg

                if w in negation:
                    negflag = 1

        elif flag == 2:   # for bigram extraction
            negflag = 0
            neg = '_NEG'
            for w, pos in zip(words, pos_tags):
                if pos in posStop:
                    negflag = 0
                if pos in posAdd and negflag == 1:
                    w += neg
                if w in negation:
                    negflag = 1

        return words

    def lex_score(self, words, lex):
        score = {}
        pos_scores = [0]
        neg_scores = [0]
        for w in words:
            if w in lex.keys():
                if lex[w] > 0:
                    pos_scores.append(lex[w])
                else:
                    neg_scores.append(lex[w])

        score['num_pos_' + self.lex_name] = len(pos_scores) - 1
        score['num_neg_' + self.lex_name] = len(neg_scores) - 1
        score['sum_pos_' + self.lex_name] = sum(pos_scores)
        score['sum_neg_' + self.lex_name] = sum(neg_scores)
        score['sum_score_' + self.lex_name] = score['sum_pos_' + self.lex_name] + score['sum_neg_' + self.lex_name]
        score['max_pos_' + self.lex_name] = max(pos_scores)
        score['min_neg_' + self.lex_name] = min(neg_scores)

        return score

    def _get_uni_scores(self):
        uni_scores = []
        for ws, pts in zip(self.words, self.pos_tags):
            new_ws = self.parse_lex(ws, pts, 1)
            uni_scores.append(self.lex_score(new_ws, self.uni_lex))

        return uni_scores

    def _get_bi_scores(self):
        bi_scores = []
        for ws, pts in zip(self.words, self.pos_tags):
            bi_words = []
            for i in range(len(ws) - 1):
                bi_words.append(ws[i] + ' ' + ws[i+1])
            bi_words = self.parse_lex(bi_words, pts, 2)
            bi_scores.append(self.lex_score(bi_words, self.bi_lex))

        return bi_scores

    def get_score(self, flag=1):
        result_dict = self._get_uni_scores()
        # print(result_dict[0])
        if flag == 2:
            bi_scores = self._get_bi_scores()
            for us, bs in zip(result_dict, bi_scores):
                for key in us.keys():
                    us[key] += bs[key]
        # print(result_dict[0])

        return result_dict


###################################################################################################
class NRCHTAG(Lexicons):
    def __init__(self, words, pos_tags):
        self.uni_path = '/Lexicon/Large-coverage Automatic Tweet Sentiment Lexicons/NRC-Hashtag-Sentiment-AffLexNegLex-v1.0/HS-AFFLEX-NEGLEX-unigrams.txt'
        self.bi_path = '/Lexicon/Large-coverage Automatic Tweet Sentiment Lexicons/NRC-Hashtag-Sentiment-AffLexNegLex-v1.0/HS-AFFLEX-NEGLEX-bigrams.txt'
        self.lex_name = 'NRCHTAG'
        super().__init__(self.lex_name, words, pos_tags, uni_path=self.uni_path, bi_path=self.bi_path)


class S140(Lexicons):
    def __init__(self, words, pos_tags):
        self.uni_path = '/Lexicon/Large-coverage Automatic Tweet Sentiment Lexicons/Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt'
        self.bi_path = '/Lexicon/Large-coverage Automatic Tweet Sentiment Lexicons/Sentiment140-Lexicon-v0.1/bigrams-pmilexicon.txt'
        self.lex_name = 'S140'
        super().__init__(self.lex_name, words, pos_tags, uni_path=self.uni_path, bi_path=self.bi_path)


class NRCEMO(Lexicons):
    def __init__(self, words, pos_tags):
        self.pos_path = '/Lexicon/Manually Created Sentiment Lexicons/NRC-Emotion-Lexicon-v0.92/NRCEmotionsPositive.txt'
        self.neg_path = '/Lexicon/Manually Created Sentiment Lexicons/NRC-Emotion-Lexicon-v0.92/NRCEmotionsNegative.txt'
        self.lex_name = 'NRCEMO'
        super().__init__(self.lex_name, words, pos_tags, pos_path=self.pos_path, neg_path=self.neg_path)


class BINGLIU(Lexicons):
    def __init__(self, words, pos_tags):
        self.pos_path = '/Lexicon/Manually Created Sentiment Lexicons/HuAndLiu/HuLiuPositive.txt'
        self.neg_path = '/Lexicon/Manually Created Sentiment Lexicons/HuAndLiu/HuLiuNegative.txt'
        self.lex_name = 'BINGLIU'
        super().__init__(self.lex_name, words, pos_tags, pos_path=self.pos_path, neg_path=self.neg_path)


class MPQA(Lexicons):
    def __init__(self, words, pos_tags):
        self.pos_path = '/Lexicon/Manually Created Sentiment Lexicons/subjectivity_clues_hltemnlp05/SubjPositive.txt'
        self.neg_path = '/Lexicon/Manually Created Sentiment Lexicons/subjectivity_clues_hltemnlp05/SubjNegative.txt'
        self.lex_name = 'MPQA'
        super().__init__(self.lex_name, words, pos_tags, pos_path=self.pos_path, neg_path=self.neg_path)


class MPQA_DCU(Lexicons):
    def __init__(self, words, pos_tags):
        self.uni_path = '/Lexicon/Manually Created Sentiment Lexicons/subjectivity_clues_hltemnlp05/DCU_MPQA.txt'
        self.lex_name = 'MPQA_DCU'
        super().__init__(self.lex_name, words, pos_tags, uni_path=self.uni_path)


class AMAZON(Lexicons):
    def __init__(self, words, pos_tags):
        self.uni_path = '/Lexicon/Specific Generated Lexicons/Amazon-laptop-electronics-reviews/Amazon-laptops-electronics-reviews-AFFLEX-NEGLEX-unigrams.txt'
        self.bi_path = '/Lexicon/Specific Generated Lexicons/Amazon-laptop-electronics-reviews/Amazon-laptops-electronics-reviews-AFFLEX-NEGLEX-bigrams.txt'
        self.lex_name = 'AMAZON'
        super().__init__(self.lex_name, words, pos_tags, uni_path=self.uni_path, bi_path=self.bi_path)


class YELP(Lexicons):
    def __init__(self, words, pos_tags):
        self.uni_path = '/Lexicon/Specific Generated Lexicons/Yelp-restaurant-reviews/Yelp-restaurant-reviews-AFFLEX-NEGLEX-unigrams.txt'
        self.bi_path = '/Lexicon/Specific Generated Lexicons/Yelp-restaurant-reviews/Yelp-restaurant-reviews-AFFLEX-NEGLEX-bigrams.txt'
        self.lex_name = 'YELP'
        super().__init__(self.lex_name, words, pos_tags, uni_path=self.uni_path, bi_path=self.bi_path)

###################################################################################################

class LexiconFeatureExtractor(object):

    def __init__(self, words, tags):
        self.nrchtag = NRCHTAG(words, tags)
        self.s140 = S140(words, tags)
        self.nrcemo = NRCEMO(words, tags)
        self.bingliu = BINGLIU(words, tags)
        self.mpqa = MPQA(words, tags)
        self.mpqa_dcu = MPQA_DCU(words, tags)
        self.amazon = AMAZON(words, tags)
        self.yelp = YELP(words, tags)
        self.vectors = self.generate_vector()

    def _generate_vector(self, lexicon2use):
        feature_names = {}
        result_vectors = []
        # fill the feature names
        for i, key in enumerate(lexicon2use.scores[0].keys()):
            feature_names[key] = i
        for score in lexicon2use.scores:
            tmp_vectors = [0 for _ in score.keys()]
            for key in score.keys():
                tmp_vectors[feature_names[key]] = score[key]

            result_vectors.append(tmp_vectors)
        result_vectors = np.asarray(result_vectors)

        return result_vectors

    def generate_vector(self):
        all_vectors = []
        all_vectors.append(self._generate_vector(self.nrchtag))
        all_vectors.append(self._generate_vector(self.s140))
        all_vectors.append(self._generate_vector(self.nrcemo))
        all_vectors.append(self._generate_vector(self.bingliu))
        all_vectors.append(self._generate_vector(self.mpqa))
        all_vectors.append(self._generate_vector(self.mpqa_dcu))
        all_vectors.append(self._generate_vector(self.amazon))
        all_vectors.append(self._generate_vector(self.yelp))
        all_vectors = np.asarray(all_vectors, dtype=np.float32)
        # print(all_vectors.shape)

        return np.mean(all_vectors, axis=0)
    
def lexicons_features_vectors(tokens, pos_tags, dependent_words=None):
    new_tokens = []
    new_pos_tags = []
    for words, tags in zip(tokens, pos_tags):
        new_words = []
        new_tags = []
        # tmp_dw_set = set([w for w in dw if w not in stop_words])
        for w, t in zip(words, tags):
            # if w in tmp_dw_set:
            new_words.append(w)
            new_tags.append(t)
        new_tokens.append(new_words)
        new_pos_tags.append(new_tags)
    return LexiconFeatureExtractor(new_tokens, new_pos_tags).vectors







