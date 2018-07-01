import numpy as np
import pandas as pd
from nltk import stem
from tqdm import tqdm
import regex as re
import scipy.sparse as sparse
from nltk.tokenize import RegexpTokenizer
from utils.definitions import SKIP_WORDS



class NLP(object):
    def __init__(self, datareader, stopwords=[]):
        """
        :param datareader: a Datareader object
        :param stopwords: a list of stopwords
        """
        self.stopwords = stopwords
        self.ps = stem.PorterStemmer()
        self.ls = stem.LancasterStemmer()

        train_playlists_df = datareader.get_df_train_playlists()
        test_playlists_df = datareader.get_df_test_playlists()

        concat_df = pd.concat([train_playlists_df, test_playlists_df])

        if datareader.offline():
            concat_df = concat_df.sort_values(['pid'], ascending=True)

        self.playlists = concat_df['pid'].values
        self.titles = concat_df['name'].values
        self.tokens_dict = dict()

        self.__set_params()
        self.words = list(self.tokens_dict.keys())

    def __normalize_name(self, title):
        name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', title)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    def __apply_porter_stemmer_appended(self, title):
        for word in title.split():
            title += " "+self.ps.stem(word)
        return title

    def __apply_lancaster_stemmer_appended(self, title):
        for word in title.split():
            title += " "+self.ls.stem(word)
        return title

    def __split_letters_and_numbers(self, title):
        for word in re.split('(\d+)', title):
            title += " " + word
        return title

    def __w_o_r_k_o_u_t(self,title):
        if all(len(element) == 1 for element in title.split()):
            new_title = title.replace(" ", "")
            # new_title = title+" "+title.replace(" ", "")
            return new_title
        return title

    def __hasNumbers(self, inputString):
        return bool(re.search(r'\d', inputString))


    def calculate_skip_words(self, title):
        to_readd=[]
        for skip_word in SKIP_WORDS:
            if skip_word in title:
                to_readd.append(skip_word.replace(" ",""))
        return to_readd


    def filter_title(self, title):
        new_title = title.lower()


        new_title = self.__w_o_r_k_o_u_t(new_title)

        new_title = self.__normalize_name(new_title)

        new_title = self.__w_o_r_k_o_u_t(new_title)

        # new_title = new_title.replace("_","")

        # new_title = new_title.replace("_"," ")

        new_title = self.__split_letters_and_numbers(new_title)

        new_title = self.__w_o_r_k_o_u_t(new_title)

        new_title = self.__apply_porter_stemmer_appended(new_title)

        new_title = self.__apply_lancaster_stemmer_appended(new_title)

        return new_title

    def __set_params(self):

        tokenizer = RegexpTokenizer(r'\w+')

        for i in tqdm(range(len(self.titles)), desc='Titles extraction'):
            title = self.titles[i]

            if type(title) is str:

                skip_words_to_readd = self.calculate_skip_words(title)
                new_title = self.filter_title(title)

                tokens = tokenizer.tokenize(new_title)

                tokens.extend(skip_words_to_readd)

                for token in tokens:

                    if len(token) > 1 and type(token) is str and token not in self.stopwords :

                        if token in self.tokens_dict.keys():
                            self.tokens_dict[token].add(self.playlists[i])
                        else:
                            self.tokens_dict[token] = {self.playlists[i]}

    def get_tokens_dict(self):
        return self.tokens_dict


    def get_UCM(self, data1=False):
        rows = []
        cols = []
        data = []

        for i in tqdm(range(len(self.words)), desc='Build UCM'):
            word = self.words[i]

            for p in self.tokens_dict[word]:
                rows.append(p)
                cols.append(i)
                data.append(1)

        ucm = sparse.csr_matrix((data, (rows, cols)), shape=(max(self.playlists)+1, len(self.words)))

        if data1:
            ucm.data=np.ones((len(ucm.data)))

        return ucm

