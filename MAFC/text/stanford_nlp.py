from stanfordcorenlp import StanfordCoreNLP
from properties.properties import theproperty

class StanfordNLP:

    def __init__(self, model_path= theproperty.rootpath + 'text/stanfordmodel/stanford-corenlp-full-2018-10-05'):
        try:
            self.standford_nlp = StanfordCoreNLP(model_path)
            #print("StanfordNLP Suceessful!")
        except RuntimeError:
            print("StanfordNLP crushed ...")
            exit(0)

    def direction_dependent(self, temp_dict, word, n):
        selected_words = []
        if word not in temp_dict.keys():
            return []
        else:
            tmp_list = temp_dict[word]
            selected_words.extend(tmp_list)
            if n > 1:
                for w in tmp_list:
                    selected_words.extend(self.direction_dependent(temp_dict, w, n - 1))

        return selected_words

    def get_dependent_words(self, words, pos_tags, text, n=2, window_size=0):
        # locate the word index of `word`
        idx = words.index('##')
        dependent_results = self.dependent_parse(text)
        in_dict = {}
        out_dict = {}
        for dr in dependent_results:
            # print(dr[0])
            src_wid = dr[1]    # source wid
            tag_wid = dr[2]    # target wid
            out_dict.setdefault(src_wid, [])
            in_dict.setdefault(tag_wid, [])

            out_dict[src_wid].append(tag_wid)
            in_dict[tag_wid].append(src_wid)

        forwards = self.direction_dependent(out_dict, idx + 1, n)
        backwards = self.direction_dependent(in_dict, idx + 1, n)

        result = []
        result.extend(forwards)
        result.extend(backwards)

        # add window-size words
        if window_size != 0:
            # right side
            for i in range(idx + 2, idx + 2 + window_size, 1):
                if i > len(words):
                    break
                result.append(i)
            for i in range(idx + 1 - window_size, idx + 1, 1):
                if i > 1:
                    result.append(i)
        result = list(set(result))
        result.sort()

        #print("!!!!!!!--->> " + " ".join(pos_tags))
        return [words[i-1] for i in result], [pos_tags[i-1] for i in result], dependent_results

    def tokenize(self, text, stop_words=[]):
        stop_words.append('##')
        stop_words = set(stop_words)
        words = [x for x in self.standford_nlp.word_tokenize(text) if x not in stop_words]

        return words

    def pos_tag(self, text):
        word_pos = self.standford_nlp.pos_tag(text)
        #print(word_pos)
        words = [x[0] for x in word_pos]
        tags = [x[1] for x in word_pos]

        return words, tags

    def dependent_parse(self, text):
        return self.standford_nlp.dependency_parse(text)
