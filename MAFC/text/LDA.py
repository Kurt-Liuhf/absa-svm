from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string 
import gensim
import pandas as pd
from gensim import corpora
class LDA:
    def __init__(self,data):
        self.stop = set(stopwords.words('english'))
        self.stop.update(("to","cc","subject","http","from","sent", "ect", "u", "fwd", "www", "com"))
        self.exclude = set(string.punctuation)
        self.lemma = WordNetLemmatizer()
        self.cleandata = [self.clean(text) for text in data]
        self.dictionary = corpora.Dictionary(self.cleandata)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.cleandata]
        
        self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=5, id2word=self.dictionary, passes=5)
        self.topics = self.ldamodel.print_topics(num_words=5)
        
    def clean(self,text):
        text = str(text)
        text = text.rstrip()
        stop_free = " ".join([i for i in text.lower().split() if((i not in self.stop) and (not i.isdigit()))])
        punc_free = ''.join(i for i in stop_free if i not in self.exclude)
        normalized = " ".join(self.lemma.lemmatize(i) for i in punc_free.split())  #词形还原    
        return normalized
    
    def get_topic_details(self):
        topic_details_df = pd.DataFrame()
        for i, row in enumerate(self.ldamodel[self.corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            #print(i,row)
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = self.ldamodel.show_topic(topic_num)
                    #print(wp)
                    topic_details_df = topic_details_df.append(pd.Series([topic_num, prop_topic]), ignore_index=True)
        topic_details_df.columns = ['Dominant_Topic', '% Score']
        return topic_details_df 
        