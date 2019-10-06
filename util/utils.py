"""
Created on 2019/3/2 15:58

@author: zhouweixin
@note: 工具
"""
import os
import errno
import numpy as np
import torch
import json
import pandas as pd
from core import config


def create_dir(path):
    """
    创建文件夹
    :param path:
    :return:
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


class Logger(object):
    def __init__(self, output_name, hyperparas=None):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            create_dir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

        if hyperparas is not None:
            self.write('=============hyperparas=============')
            for key, value in hyperparas.items():
                self.write('{}: {}'.format(key, value))
            self.write('====================================')
            self.write()

    def append(self, key, value):
        self.infos.setdefault(key, []).append(value)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, values in self.infos.items():
            msgs.append('%s %.6f' % (key, np.mean(values)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg=''):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)


def l2_loss(x, y=None):
    if y is None:
        return torch.sum(x ** 2) / 2
    else:
        return torch.sum((x - y) ** 2) / 2


def save_review_att(user_id, item_id, user_rids, item_rids, user_reviews_att, item_reviews_att, rating, user_word_att,
                    item_word_att):
    data = {'user_id': user_id.cpu().numpy().tolist(),
            'item_id': item_id.cpu().numpy().tolist(),
            'user_rids': user_rids.cpu().numpy().tolist(),
            'item_rids': item_rids.cpu().numpy().tolist(),
            'user_reviews_att': user_reviews_att.data.cpu().numpy().tolist(),
            'item_reviews_att': item_reviews_att.data.cpu().numpy().tolist(),
            'rating': rating.cpu().numpy().tolist(),
            'user_word_att': user_word_att.data.cpu().numpy().tolist(),
            'item_word_att': item_word_att.data.cpu().numpy().tolist()}

    data = json.dumps(data)

    file = os.path.join(config.output_path, 'review_attention.txt')
    with open(file, 'w+') as f:
        f.write(data + '\n')


def load_review_att_to_json():
    datas = []

    file = os.path.join(config.output_path, 'review_attention.txt')
    with open(file, 'r') as f:
        while True:
            data = f.readline()
            if not data:
                break

            data = json.loads(data)

            user_id = data['user_id']
            item_id = data['item_id']
            user_rids = data['user_rids']
            item_rids = data['item_rids']
            user_reviews_att = data['user_reviews_att']
            item_reviews_att = data['item_reviews_att']
            rating = data['rating']
            user_word_att = data['user_word_att']
            item_word_att = data['item_word_att']
            for i in range(len(rating)):
                datas.append({'user_id': user_id[i],
                              'item_id': item_id[i],
                              'user_rids': user_rids[i],
                              'item_rids': item_rids[i],
                              'user_reviews_att': user_reviews_att[i],
                              'item_reviews_att': item_reviews_att[i],
                              'rating': rating[i],
                              'user_word_att': user_word_att[i],
                              'item_word_att': item_word_att[i]})
    json.dump(datas, open(os.path.join(config.output_path, 'review_attention.json'), 'w'))


def load_review_att(idx2word):
    user_review_path = '../data/toy_game/user_reviews'
    item_review_path = '../data/toy_game/item_reviews'
    user_review_token_path = '../data/toy_game/user_reviews_token'
    item_review_token_path = '../data/toy_game/item_reviews_token'

    datas = json.load(open('../data/review_attention.json', 'r'))
    for data in datas:
        user_id = data['user_id']
        item_id = data['item_id']
        user_rids = data['user_rids']
        item_rids = data['item_rids']
        rating = data['rating']
        user_reviews_att = data['user_reviews_att']
        item_reviews_att = data['item_reviews_att']
        user_word_att = data['user_word_att']
        item_word_att = data['item_word_att']

        id2review = {}
        item_reviews = json.load(open(os.path.join(item_review_path, str(item_id[0]) + '.json'), 'r'))
        for item_review in item_reviews:
            review = item_review['review']
            id = item_review['id']
            id2review[id] = review

        reviews = [id2review.setdefault(id, '') for id in item_rids]

        id2review_token = {}
        item_reviews_token = json.load(open(os.path.join(item_review_token_path, str(item_id[0]) + '.json'), 'r'))
        reviews_token = item_reviews_token['reviews']
        for i, token in enumerate(reviews_token):
            words = []
            for idx in token:
                if idx < len(idx2word):
                    words.append(idx2word[idx])
                else:
                    words.append(-1)
            reviews_token[i] = words

        reviews_token = [reviews_token[i] for i in range(len(item_rids))]

        df = pd.DataFrame({'item_id': item_id * len(item_reviews_att), 'rating': rating * len(item_reviews_att),
                           'item_reviews_att': np.squeeze(item_reviews_att), 'reviews': reviews,
                           'reviews_token': reviews_token, 'item_word_att': item_word_att})
        df = df.sort_values(by=['item_reviews_att'], ascending=False)
        df.to_excel('../data/item_review_att/' + str(item_id[0]) + '.xlsx')

# load_review_att_to_json()
