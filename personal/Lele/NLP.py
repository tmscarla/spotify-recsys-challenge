import numpy as np
import pandas as pd
from nltk import stem
from tqdm import tqdm
# import emoji
import regex as re
import scipy.sparse as sparse
from nltk.tokenize import RegexpTokenizer
from difflib import SequenceMatcher
from difflib import get_close_matches
from collections import OrderedDict
from utils.datareader import Datareader
from nltk.stem.lancaster import LancasterStemmer
from utils.definitions import SKIP_WORDS



class NLP2(object):
    def __init__(self, datareader, stopwords,norm,work,split, skip_words,
                 date,porter,porter2,lanca,lanca2):
        """
        :param datareader: a Datareader object
        :param stopwords: a list of stopwords
        """
        self.norm = norm
        self.work = work
        self.split = split
        self.skip_words = skip_words
        self.date = date

        self.porter = porter
        self.porter2 = porter2
        self.lanca = lanca
        self.lanca2 = lanca2

        self.ps = stem.PorterStemmer()
        self.ls = stem.LancasterStemmer()

        train_playlists_df = datareader.get_df_train_playlists()
        test_playlists_df = datareader.get_df_test_playlists()
        concat_df = pd.concat([train_playlists_df, test_playlists_df])
        concat_df = concat_df.sort_values(['pid'], ascending=True)

        self.stopwords = stopwords
        self.titles = concat_df['name'].as_matrix()
        self.tokens_dict = dict()

        self.__set_params()
        self.words = list(self.tokens_dict.keys())

    def __normalize_name(self, title):
        name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', title)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    def __apply_porter_stemmer(self, title):
        new_title = ""
        for word in title.split():
            new_title += self.ps.stem(word) + " "
        return new_title

    def __apply_porter_stemmer_appended(self, title):
        for word in title.split():
            title += " "+self.ps.stem(word)
        return title

    def __apply_lancaster_stemmer_appended(self, title):
        for word in title.split():
            title += " "+self.ls.stem(word)
        return title

    def __apply_lancaster_stemmer(self, title):
        new_title = ""
        for word in title.split():
            new_title += self.ls.stem(word) + " "
        return new_title

    def __filter_date(self, title):

        # 2k10 . 2k15
        for i in range(0, 20):
            title = re.sub(r"2k" + str(i).zfill(2), '20' + str(i).zfill(2), title)

        # date precise tipo '18 '15 '17
        for i in range(10, 20):
            title = re.sub(r"'" + str(i), '20' + str(i), title)

        # date tipo 00s 2000s
        for i in range(0, 10):
            # CASO 1930's -> 1990's /   30's -> 90's  /  90s
            if i > 2:
                match = re.search(
                    "(19" + str(i) + "0's)|(^" + str(i) + "0's)|((?<![0-9])" + str(i) + "0s)|((?<![0-9])" + str(
                        i) + "0's)", title)
                if match:
                    title = title[0:match.span()[0]] + " 19" + str(i) + "0s " + title[match.span()[1]:len(title)]
            # caso 2000's 00's  /// .  10's / . 10s
            else:
                match = re.search(
                    "(20" + str(i) + "0's)|(^" + str(i) + "0's)|((?<![0-9])" + str(i) + "0s)|((?<![0-9])" + str(
                        i) + "0's)", title)
                if match:
                    title = title[0:match.span()[0]] + " 20" + str(i) + "0s " + title[match.span()[1]:len(title)]

            title = re.sub(' +', ' ', title)
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

    def print_some_checkups(self, title):
        lista_strana = [element for element in title.split() if len(element) == 1]
        if len(lista_strana) > 3:
            print(title, ">>>>>daguardare. tokens cortihehehehe", lista_strana)


    def filter_title(self, title, norm=False,work=False,split=False,date=False,porter2=False,lanca2=False,
                      underscore1=False, underscore2=False):
        new_title = title.lower()
        # check = new_title
        if norm:
            new_title = self.__normalize_name(new_title)
        if work:
            new_title = self.__w_o_r_k_o_u_t(new_title)
        if underscore1:
            new_title = new_title.replace("_","")
        if underscore2:
            new_title = new_title.replace("_"," ")
        if split:
            new_title = self.__split_letters_and_numbers(new_title)
        if date:
            if self.__hasNumbers(new_title):
                new_title = self.__filter_date(new_title)
                new_title = self.__filter_date(new_title)
        if porter2:
            new_title = self.__apply_porter_stemmer_appended(new_title)
        if lanca2:
            new_title = self.__apply_lancaster_stemmer_appended(new_title)
        return new_title

    def __set_params(self):

        tokenizer = RegexpTokenizer(r'\w+')

        for i in tqdm(range(len(self.titles)), desc='Titles extraction'):
            title = self.titles[i]

            if type(title) is str:

                skip_words_to_readd = self.calculate_skip_words(title)
                new_title = self.filter_title(title, norm=self.norm,work=self.work,split=self.split,
                                          date=self.date,porter2= self.porter2,lanca2=self.lanca2)

                # self.print_some_checkups(new_title)

                tokens = tokenizer.tokenize(new_title)
                if self.skip_words:
                    tokens.extend(skip_words_to_readd)


                for token in tokens:
                    if token not in self.stopwords and len(token) > 1 and type(token) is str:
                        s = token
                        if s not in SKIP_WORDS:
                            if self.porter:
                                s = self.ps.stem(s)
                            if self.lanca:
                                s = self.ls.stem(s)

                        if s in self.tokens_dict.keys():
                            self.tokens_dict[s].add(i)
                        else:
                            self.tokens_dict[s] = {i}
                    # elif type(token) is not str:
                    #     print(title,">>>>",new_title,">>>>>",token,"(is not str)")

    def get_UCM(self, data1):
        rows = []
        cols = []
        data = []

        for i in tqdm(range(len(self.words)), desc='Build UCM'):
            word = self.words[i]

            for p in self.tokens_dict[word]:
                rows.append(p)
                cols.append(i)
                data.append(1)

        ucm = sparse.csr_matrix((data, (rows, cols)), shape=(len(self.titles), len(self.words)))

        if data1:
            ucm.data=np.ones((len(ucm.data)))

        return ucm

