"""
Created on 2019/3/2 15:58

@author: zhouweixin
@note: 加载数据和分割数据集
"""
import json
import os
import re
import numpy as np
import pandas as pd
from core import config
from core.dataset import Dictionary


def get_id2num(data, id):
    """
    按user和item分组, 求每一个组的个数
    :param data:
    :param id:
    :return:
    """
    id2num = data[[id, 'ratings']].groupby(id, as_index=False)
    return id2num.size()


def encode_user_item(data, user2idx, item2idx):
    """
    给user和item编码
    :param data:
    :param user2idx:
    :param item2idx:
    :return:
    """
    data['user_ids'] = pd.Series(map(lambda x: user2idx[x], data['user_ids']))
    data['item_ids'] = pd.Series(map(lambda x: item2idx[x], data['item_ids']))
    return data


def load_data(data_path_file=config.data_path_file):
    """
    加载数据
    :param data_path_file:
    :return:
    """

    user_ids = []
    item_ids = []
    ratings = []
    reviews = []

    with open(data_path_file) as f:
        line = f.readline()
        while line:
            js = json.loads(line)
            line = f.readline()

            # 跳过没有id的user和item
            if str(js['reviewerID']) == 'unknown':
                print('unknown user_id')
            if str(js['asin']) == 'unknown':
                print('unknown item_id')

            user_ids.append(js['reviewerID'])
            item_ids.append(js['asin'])
            ratings.append(js['overall'])
            reviews.append(js['reviewText'])

    print('从文件中加载完成, 总数: %d' % len(ratings))

    # 转储成DataFrame
    data = pd.DataFrame({'user_ids': pd.Series(user_ids),
                         'item_ids': pd.Series(item_ids),
                         'ratings': pd.Series(ratings),
                         'reviews': pd.Series(reviews)})[['user_ids', 'item_ids', 'ratings', 'reviews']]

    # 统计每个user和item的评分数(每个user评论的个数, 每个item收到评论的个数)
    user_id2num = get_id2num(data, 'user_ids')
    item_id2num = get_id2num(data, 'item_ids')

    # 统计user和item的id
    unique_user_ids = user_id2num.index
    unique_item_ids = item_id2num.index

    # user和item编码
    user2idx = dict((user_id, i) for i, user_id in enumerate(unique_user_ids))
    item2idx = dict((item_id, i) for i, item_id in enumerate(unique_item_ids))

    # 编码原始数据
    data = encode_user_item(data, user2idx, item2idx)

    # 保存user_num和item_num
    user_num = len(unique_user_ids)
    item_num = len(unique_item_ids)
    num = {"user_num": user_num, "item_num": item_num}
    json.dump(num, open(os.path.join(config.output_path, 'user_item_num.json'), 'w'))
    print('保存: user_item_num.json')

    return data


def split_train_val_test(data, data_ratio = config.data_ratio, ratio=config.train_val_test_ratio, output_path=config.output_path):
    """
    分割训练集, 验证集和测试集
    data 数据
    ratio 训练集, 验证集和测试的比例
    """

    # 0.select data set
    num_ratings = data.shape[0]
    random_idx = np.random.choice(num_ratings, size=int(data_ratio * num_ratings), replace=False)
    train_idx = np.zeros(num_ratings, dtype=bool)
    train_idx[random_idx] = True
    data = data[train_idx]

    # 1.分割训练集
    num_ratings = data.shape[0]
    random_idx = np.random.choice(num_ratings, size=int(ratio[0] * num_ratings), replace=False)
    train_idx = np.zeros(num_ratings, dtype=bool)
    train_idx[random_idx] = True

    train_data = data[train_idx]  # 训练集
    val_test_data = data[~train_idx]

    temp_train_data = train_data
    temp_val_test_data = val_test_data

    # 2.分割验证集和测试集
    num_ratings = val_test_data.shape[0]
    random_idx = np.random.choice(num_ratings, size=int(ratio[1] / (ratio[1] + ratio[2]) * num_ratings), replace=False)
    val_idx = np.zeros(num_ratings, dtype=bool)
    val_idx[random_idx] = True

    val_data = val_test_data[val_idx]  # 验证集
    test_data = val_test_data[~val_idx]  # 测试集

    train_data = train_data[['user_ids', 'item_ids', 'ratings']]
    val_data = val_data[['user_ids', 'item_ids', 'ratings']]
    test_data = test_data[['user_ids', 'item_ids', 'ratings']]

    # 保存为csv文件
    train_path = os.path.join(output_path, 'train.csv')
    val_path = os.path.join(output_path, 'val.csv')
    test_path = os.path.join(output_path, 'test.csv')

    header = ['user_ids', 'item_ids', 'ratings']
    train_data.to_csv(train_path, index=False, header=header)
    val_data.to_csv(val_path, index=False, header=header)
    test_data.to_csv(test_path, index=False, header=header)

    print("训练集保存%d: %s" % (train_data.shape[0], train_path))
    print("验证集集保存%d: %s" % (val_data.shape[0], val_path))
    print("测试集保存%d: %s" % (test_data.shape[0], test_path))
    return temp_train_data, temp_val_test_data, train_data, val_data, test_data


def split_review(temp_train_data, temp_val_test_data, output_path=config.output_path):
    """
    分离评论
    :param temp_train_data: 训练集
    :param temp_val_test_data: 验证集和测试集
    :param output_path:
    :return:
    """

    user_review_rids = {}
    item_review_rids = {}

    # 只有训练集需要评论
    for (user_id, item_id, rating, review) in temp_train_data.values:
        user_review_rids.setdefault(user_id, []).append({'review': review, 'id':item_id})
        item_review_rids.setdefault(item_id, []).append({'review': review, 'id':user_id})

    # 验证集和测试集不需要评论
    for (user_id, item_id, rating, review) in temp_val_test_data.values:
        if user_id not in user_review_rids:
            user_review_rids.setdefault(user_id, [])
        else:
            user_review_rids.setdefault(user_id, []).append({'review': review, 'id': item_id})

        if item_id not in item_review_rids:
            item_review_rids.setdefault(item_id, [])
        else:
            item_review_rids.setdefault(item_id, []).append({'review': review, 'id': user_id})

    # 计算评论数量和评论长度, 保存评论数据
    user_review_num, user_review_len = compute_reviews_num_len(user_review_rids, 'user')
    item_review_num, item_review_len = compute_reviews_num_len(item_review_rids, 'item')

    review_num_len = {'user_review_num': user_review_num,
                      'user_review_len': user_review_len,
                      'item_review_num': item_review_num,
                      'item_review_len': item_review_len}
    json.dump(review_num_len, open(os.path.join(output_path, 'review_num_len.json'), 'w'))
    print('保存: user_reviews')
    print('保存: item_reviews')
    print('保存: review_num_len.json')


def compute_reviews_num_len(id_review_rids, type, output_path=config.output_path):
    """
    计算评论数量和评论长度, 保存评论数据
    :param id_review_rids:
    :param type:
    :param output_path:
    :return:
    """
    assert type in ['user', 'item']

    path = os.path.join(output_path, type + "_reviews")
    os.mkdir(path) if not os.path.exists(path) else True

    dictionary = Dictionary()

    review_nums = []
    review_lens = []
    for id, review_rids in id_review_rids.items():
        # 保存评论数据
        json.dump(review_rids, open(os.path.join(path, str(id) + ".json"), 'w'))

        review_nums.append(len(review_rids))
        for review_rid in review_rids:
            review = review_rid['review']
            # review = clean_str(review).split()
            review = dictionary.tokenize(review, True)
            review_lens.append(len(review))

    review_nums = np.sort(np.array(review_nums))
    review_lens = np.sort(np.array(review_lens))

    review_num = int(review_nums[int(config.review_num_ratio * len(review_nums)) - 1])
    review_len = int(review_lens[int(config.review_len_ratio * len(review_lens)) - 1])
    return review_num, review_len


def clean_str(string):
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


data = load_data()
temp_train_data, temp_val_test_data, train_data, val_data, test_data = split_train_val_test(data)
split_review(temp_train_data, temp_val_test_data)
print("完成")
