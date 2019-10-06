"""
Created on 2019/3/2 15:58

@author: zhouweixin
@note: 自定义数据集
"""

import json
import os
import pickle
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from core import config
from util import utils


def clean_str1(string):
    """
    清楚特殊字符
    :param string:
    :return:
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()
    return string


def clean_str2(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


class Dictionary():
    """
    字典：包含两个成员变量（idx2word, word2idx）
    """

    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}

        if idx2word is None:
            idx2word = []

        # self.stopwords = list(set(stopwords.words('english')))
        self.stopwords = self.load_stopwords(config.stopwords)
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def padding_idx(self):
        return len(self.word2idx)

    @property
    def ntoken(self):
        return len(self.word2idx)

    def tokenize(self, text, is_add_word=False):
        """
        分词
        :param text: 文本数据
        :param is_add_word: 是否添加词汇
        :return:
        """

        # clean
        text = clean_str2(text)
        # tokenize
        words = text.split()
        # clean stopwords
        words = [w for w in words if w not in self.stopwords]

        tokens = []
        if is_add_word:
            for word in words:
                tokens.append(self.add_word(word))
        else:
            for word in words:
                tokens.append(self.word2idx[word])

        return tokens

    def load_stopwords(self, stopwords_file):
        """
        加载停止词
        :param stopwords_file: 停止词文件
        :return:
        """
        stopwords = []
        with open(stopwords_file) as f:
            for word in f:
                stopwords.append(word.strip())

        return stopwords

    def add_word(self, word):
        """
        添加词汇
        :param word: 单词
        :return:
        """
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

        return self.word2idx[word]

    def dump_to_file(self, path):
        """
        存储字典
        :param path:
        :return:
        """
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path=config.dictionary_path_file):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        return cls(word2idx, idx2word)

    def tokenize_padding(self, review_rids, review_num, review_len, padding_word_idx, padding_id):
        """
        分词填充
        :param review_rids:
        :param review_num:
        :param review_len:
        :param padding_word_idx:
        :param padding_id:
        :return:
        """
        # padding word
        for index, review_rid in enumerate(review_rids):
            review = review_rid['review']
            # 分词
            review = self.tokenize(review, is_add_word=False)

            if review_len < len(review):
                review = review[:review_len]
            else:
                review = [padding_word_idx] * (review_len - len(review)) + review

            assert len(review) == review_len
            review_rids[index]['review'] = review

        # padding sentence
        if review_num < len(review_rids):
            review_rids = review_rids[:review_num]
        else:
            review_rids = [{'review': [padding_word_idx] * review_len, 'id': padding_id}] * (
                review_num - len(review_rids)) + review_rids

        # 分离评论内容和评论对象
        reviews = []
        rids = []
        for review_rid in review_rids:
            reviews.append(review_rid['review'])
            rids.append(review_rid['id'])

        return reviews, rids

    def process_review(self):
        """
        预处理评论
        :return:
        """
        # 创建目录
        utils.create_dir(os.path.join(config.data_root, config.data_type, 'user_reviews_token'))
        utils.create_dir(os.path.join(config.data_root, config.data_type, 'item_reviews_token'))

        # 加载user_num和item_num
        user_item_num = json.load(open(os.path.join(config.output_path, 'user_item_num.json'), 'r'))

        # 加载user_review_num和item_review_num
        review_num_len = json.load(open(os.path.join(config.output_path, 'review_num_len.json'), 'r'))

        for t in ['user', 'item']:
            t1 = 'item' if t == 'user' else 'user'
            path = os.path.join(config.data_root, config.data_type, t + '_reviews')
            files = os.listdir(path)
            for file in files:
                # 1.加载评论文件
                review_rids = json.load(open(os.path.join(path, file), 'r'))

                # 2.分词, 填充
                reviews, rids = self.tokenize_padding(review_rids,
                                                      review_num_len[t + '_review_num'],
                                                      review_num_len[t + '_review_len'],
                                                      self.padding_idx,
                                                      user_item_num[t1 + '_num'])
                # 3.保存
                json.dump({'reviews': reviews, 'rids': rids},
                          open(os.path.join(config.data_root, config.data_type, t + '_reviews_token', file), 'w'))


class NARREDataset(Dataset):
    def __init__(self, name, dictionary, data_type=config.data_type):
        """
        NARRE模型数据集
        :param name: 数据集类型: 训练集, 验证集, 测试集
        :param dictionary: 字典
        :param data_type: 数据类型
        """
        super(NARREDataset, self).__init__()
        assert name in ['train', 'test', 'val']

        print("loading %s dataset" % name)
        self.root = os.path.join(config.data_root, data_type)

        self.dictionary = dictionary
        self.name = name

        # 加载user_num和item_num
        user_item_num = json.load(open(os.path.join(config.output_path, 'user_item_num.json'), 'r'))
        self.user_num = user_item_num['user_num']
        self.item_num = user_item_num['item_num']

        # 加载user_review_num和item_review_num
        review_num_len = json.load(open(os.path.join(config.output_path, 'review_num_len.json'), 'r'))
        self.user_review_num = review_num_len['user_review_num']
        self.item_review_num = review_num_len['item_review_num']
        self.user_review_len = review_num_len['user_review_len']
        self.item_review_len = review_num_len['item_review_len']

        # 加载评分数据
        self.user_ids, self.item_ids, self.ratings = self.load_ratings(os.path.join(self.root, '%s.csv' % name))

    def load_ratings(self, filename):
        """
        读取评分数据
        :param filename:
        :return:
        """
        pd_data = pd.read_csv(filename)
        user_ids = np.array(list(pd_data['user_ids']), dtype=np.int32)
        item_ids = np.array(list(pd_data['item_ids']), dtype=np.int32)
        ratings = np.array(list(pd_data['ratings']), dtype=np.float32)
        assert len(user_ids) == len(item_ids) and len(user_ids) == len(ratings)

        print('reading %s' % filename)
        return user_ids, item_ids, ratings

    def tokenize_padding(self, review_rids, review_num, review_len, padding_word_idx, padding_id):
        """
        创建数据集
        :param review_rids:
        :param review_len:
        :return:
        """
        # padding word
        for index, review_rid in enumerate(review_rids):
            review = review_rid['review']
            # 分词
            review = self.dictionary.tokenize(review, is_add_word=False)

            if review_len < len(review):
                review = review[:review_len]
            else:
                review = [padding_word_idx] * (review_len - len(review)) + review

            assert len(review) == review_len
            review_rids[index]['review'] = review

        # padding sentence
        if review_num < len(review_rids):
            review_rids = review_rids[:review_num]
        else:
            review_rids = [{'review': [padding_word_idx] * review_len, 'id': padding_id}] * (
                review_num - len(review_rids)) + review_rids

        # 分离评论内容和评论对象
        reviews = []
        rids = []
        for review_rid in review_rids:
            reviews.append(review_rid['review'])
            rids.append(review_rid['id'])

        return reviews, rids

    def tensorize(self, user_id, item_id, user_reviews, item_reviews, user_rids, item_rids, rating):
        """
        张量化
        :param user_id:
        :param item_id:
        :param user_review:
        :param item_review:
        :param rating:
        :return:
        """

        user_id = torch.LongTensor([user_id])
        item_id = torch.LongTensor([item_id])
        rating = torch.FloatTensor([rating])
        user_reviews = torch.LongTensor(user_reviews)
        item_reviews = torch.LongTensor(item_reviews)
        user_rids = torch.LongTensor(user_rids)
        item_rids = torch.LongTensor(item_rids)
        return user_id, item_id, user_reviews, item_reviews, user_rids, item_rids, rating

    def load_review(self, id, t, output_path=config.output_path):
        """
        根据id加载评论数据
        :param id:
        :param t:
        :param output_path:
        :return:
        """
        assert t in ['user', 'item']

        reviews_path = os.path.join(output_path, t + '_reviews')
        reviews_filename = os.path.join(reviews_path, str(id) + '.json')
        review_rids = json.load(open(reviews_filename, 'r'))

        return review_rids

    def load_reviews_rids(self, id, t):
        """
        根据id加载评论数据
        :param id:
        :param t:
        :return:
        """
        assert t in ['user', 'item']

        filepath = os.path.join(config.data_root, config.data_type, t + '_reviews_token', str(id) + '.json')

        reviews_rids = json.load(open(filepath, 'r'))

        return reviews_rids['reviews'], reviews_rids['rids']

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        item_id = self.item_ids[index]
        rating = self.ratings[index]
        user_reviews, user_rids = self.load_reviews_rids(user_id, 'user')
        item_reviews, item_rids = self.load_reviews_rids(item_id, 'item')

        # 张量化: user_id, item_id, user_reviews, item_reviews, user_rids, item_rids, rating
        item = self.tensorize(user_id, item_id,
                              user_reviews,
                              item_reviews,
                              user_rids,
                              item_rids,
                              rating)
        return item

    def __len__(self):
        return len(self.ratings)
