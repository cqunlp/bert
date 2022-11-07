# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
FastText data preprocess
"""
import re
import html
import csv
import spacy
from sklearn.feature_extraction import FeatureHasher
from mindtext.common.data import Pad


class DataPreProcess():
    """Data preprocess"""

    def __init__(self, data_path=None,
                 max_length=None,
                 class_num=None,
                 ngram=None,
                 feature_dict=None,
                 buckets=None,
                 is_hashed=None,
                 feature_size=None,
                 is_train=True):
        self.data_path = data_path
        self.max_length = max_length
        self.class_num = class_num
        self.feature_dict = feature_dict
        self.is_hashed = is_hashed
        self.feature_size = feature_size
        self.buckets = buckets
        self.ngram = ngram
        self.text_greater = '>'
        self.text_less = '<'
        self.word2vec = dict()
        self.vec2words = dict()
        self.non_str = '\\'
        self.end_string = ['.', '?', '!']
        self.word2vec['PAD'] = 0
        self.vec2words[0] = 'PAD'
        self.word2vec['UNK'] = 1
        self.vec2words[1] = 'UNK'
        self.str_html = re.compile(r'<[^>]+>')
        self.is_train = is_train

    def common_block(self, pair_sen, spacy_nlp):
        """common block for data preprocessing"""
        label_idx = int(pair_sen[0]) - 1
        if len(pair_sen) == 3:
            src_tokens = self.input_preprocess(src_text1=pair_sen[1],
                                               src_text2=pair_sen[2],
                                               spacy_nlp=spacy_nlp,
                                               train_mode=True)
            src_tokens_length = len(src_tokens)
        elif len(pair_sen) == 2:
            src_tokens = self.input_preprocess(src_text1=pair_sen[1],
                                               src_text2=None,
                                               spacy_nlp=spacy_nlp,
                                               train_mode=True)
            src_tokens_length = len(src_tokens)
        elif len(pair_sen) == 4:
            if pair_sen[2]:
                sen_o_t = pair_sen[1] + ' ' + pair_sen[2]
            else:
                sen_o_t = pair_sen[1]
            src_tokens = self.input_preprocess(src_text1=sen_o_t,
                                               src_text2=pair_sen[3],
                                               spacy_nlp=spacy_nlp,
                                               train_mode=True)
            src_tokens_length = len(src_tokens)
        return src_tokens, src_tokens_length, label_idx

    def load(self):
        """data preprocess loader"""
        dataset_list = []
        spacy_nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger', 'ner'])
        spacy_nlp.add_pipe(spacy_nlp.create_pipe('sentencizer'))

        if self.is_train:
            with open(self.data_path, 'r', newline='', encoding='utf-8') as data_file:
                reader = csv.reader(data_file, delimiter=",", quotechar='"')
                for _, pair_sen in enumerate(reader):
                    src_tokens, src_tokens_length, label_idx = self.common_block(pair_sen=pair_sen,
                                                                                 spacy_nlp=spacy_nlp)
                    dataset_list.append([src_tokens, src_tokens_length, label_idx])
        else:
            with open(self.data_path, 'r', newline='', encoding='utf-8') as data_file:
                reader2 = csv.reader(data_file, delimiter=",", quotechar='"')
                for _, test_sen in enumerate(reader2):
                    label_idx = int(test_sen[0]) - 1
                    if len(test_sen) == 3:
                        src_tokens = self.input_preprocess(src_text1=test_sen[1],
                                                           src_text2=test_sen[2],
                                                           spacy_nlp=spacy_nlp,
                                                           train_mode=False)
                        src_tokens_length = len(src_tokens)
                    elif len(test_sen) == 2:
                        src_tokens = self.input_preprocess(src_text1=test_sen[1],
                                                           src_text2=None,
                                                           spacy_nlp=spacy_nlp,
                                                           train_mode=False)
                        src_tokens_length = len(src_tokens)
                    elif len(test_sen) == 4:
                        if test_sen[2]:
                            sen_o_t = test_sen[1] + ' ' + test_sen[2]
                        else:
                            sen_o_t = test_sen[1]
                        src_tokens = self.input_preprocess(src_text1=sen_o_t,
                                                           src_text2=test_sen[3],
                                                           spacy_nlp=spacy_nlp,
                                                           train_mode=False)
                        src_tokens_length = len(src_tokens)

                    dataset_list.append([src_tokens, src_tokens_length, label_idx])

        if self.is_hashed:
            print("Begin to Hashing Trick......")
            features_num = self.feature_size
            fh = FeatureHasher(n_features=features_num, alternate_sign=False)
            print("FeatureHasher features..", features_num)
            self.hash_trick(fh, dataset_list)
            print("Hashing Done....")

        # pad dataset
        dataset_list_length = len(dataset_list)
        for i in range(dataset_list_length):
            bucket_length = self._get_bucket_length(dataset_list[i][0], self.buckets)
            dataset_list[i][0] = Pad(bucket_length)(dataset_list[i][0])
            dataset_list[i][1] = len(dataset_list[i][0])

        example_data = []
        for idx in range(dataset_list_length):
            example_data.append({
                "src_tokens": dataset_list[idx][0],
                "src_tokens_length": dataset_list[idx][1],
                "label_idx": dataset_list[idx][2],
            })
            for key in self.feature_dict:
                if key == example_data[idx]['src_tokens_length']:
                    self.feature_dict[key].append(example_data[idx])

        if self.is_train:
            print("train vocab size is ", len(self.word2vec))

        return self.feature_dict

    def input_preprocess(self, src_text1, src_text2, spacy_nlp, train_mode):
        """data preprocess func"""
        src_text1 = src_text1.strip()
        if src_text1 and src_text1[-1] not in self.end_string:
            src_text1 = src_text1 + '.'

        if src_text2:
            src_text2 = src_text2.strip()
            sent_describe = src_text1 + ' ' + src_text2
        else:
            sent_describe = src_text1
        if self.non_str in sent_describe:
            sent_describe = sent_describe.replace(self.non_str, ' ')

        sent_describe = html.unescape(sent_describe)

        if self.text_less in sent_describe and self.text_greater in sent_describe:
            sent_describe = self.str_html.sub('', sent_describe)

        doc = spacy_nlp(sent_describe)
        bows_token = [token.text for token in doc]

        try:
            tagged_sent_desc = '<p> ' + ' </s> '.join([s.text for s in doc.sents]) + ' </p>'
        except ValueError:
            tagged_sent_desc = '<p> ' + sent_describe + ' </p>'
        doc = spacy_nlp(tagged_sent_desc)
        ngrams = self.generate_gram([token.text for token in doc], num=self.ngram)

        bo_ngrams = bows_token + ngrams

        if train_mode is True:
            for ngms in bo_ngrams:
                idx = self.word2vec.get(ngms)
                if idx is None:
                    idx = len(self.word2vec)
                    self.word2vec[ngms] = idx
                    self.vec2words[idx] = ngms

        processed_out = [self.word2vec[ng] if ng in self.word2vec else self.word2vec['UNK'] for ng in bo_ngrams]

        return processed_out

    def _get_bucket_length(self, x, bts):
        x_len = len(x)
        for index in range(1, len(bts)):
            if bts[index - 1] < x_len <= bts[index]:
                return bts[index]
        return bts[0]

    def generate_gram(self, words, num=2):

        return [' '.join(words[i: i + num]) for i in range(len(words) - num + 1)]

    def count2dict(self, lst):
        count_dict = dict()
        for m in lst:
            if str(m) in count_dict:
                count_dict[str(m)] += 1
            else:
                count_dict[str(m)] = 1
        return count_dict

    def hash_trick(self, hashing, input_data):
        trans = hashing.transform((self.count2dict(e[0]) for e in input_data))
        for htr, e in zip(trans, input_data):
            sparse2bow = list()
            for idc, d in zip(htr.indices, htr.data):
                for _ in range(int(d)):
                    sparse2bow.append(idc + 1)
            e[0] = sparse2bow

    def vocab_to_txt(self, path):
        with open(path, "w") as f:
            for k, v in self.word2vec.items():
                f.write(k + "\t" + str(v) + "\n")

    def read_vocab_txt(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
            for i, word in enumerate(lines):
                s = word.split("\t")
                self.word2vec[s[0]] = i
