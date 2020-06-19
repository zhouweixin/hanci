"""
Created on 2020/3/19 15:01

@author: zhouweixin
@note: 
"""
from core import config
import json
import random


def load_data(data_path_file=config.data_path_file):
    """
    加载数据
    :param data_path_file:
    :return:
    """

    item_id2review_info = {}

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

            item_id = js['asin']
            user_id = js['reviewerID']
            helpful = js['helpful']
            review_time = js['unixReviewTime']
            review_text = js['reviewText']

            if helpful[0] == 0:
                continue

            user_id2helpful = item_id2review_info.setdefault(item_id, [])
            user_id2helpful.append(
                {'userId': user_id, 'helpful': helpful, 'reviewTime': review_time, 'reviewLen': len(review_text)})

    datas = []
    for item_id, review_infos in item_id2review_info.items():
        max_time = 0
        max_time_user_id = 0
        max_review_len = 0
        max_review_len_user_id = 0
        max_helpful = 0
        max_helpful_user_id = 0

        user_ids = []
        for review_info in review_infos:
            user_ids.append(review_info['userId'])

            if review_info['reviewTime'] > max_time:
                max_time = review_info['reviewTime']
                max_time_user_id = review_info['userId']

            if review_info['reviewLen'] > max_review_len:
                max_review_len = review_info['reviewLen']
                max_review_len_user_id = review_info['userId']

            if review_info['helpful'][1] > max_helpful:
                max_helpful = review_info['helpful'][1]
                max_helpful_user_id = review_info['userId']

        randomIdx = random.randint(0, len(user_ids) - 1)
        datas.append({item_id:
            {'helpful': max_helpful_user_id, 'random': user_ids[randomIdx], 'latest': max_time_user_id,
             'length': max_review_len_user_id}})

    return datas


def computeRate(datas):
    total_num = len(datas)
    random_num = 0
    latest_num = 0
    length_num = 0
    for data in datas:
        d = list(data.values())[0]
        helpful = d['helpful']
        random = d['random']
        latest = d['latest']
        length = d['length']

        if helpful == random:
            random_num += 1

        if helpful == latest:
            latest_num += 1

        if helpful == length:
            length_num += 1

    latest_rate = latest_num / total_num
    random_rate = random_num / total_num
    length_rate = length_num / total_num

    print('latest_rate = %.4f, random_rate = %.4f, length_rate = %.4f' % (latest_rate, random_rate, length_rate))



datas = load_data()
computeRate(datas)
