import pandas as pd
import json
import sys
from core import config


def sample(user_ids, item_ids, ratings, reviews, num):
    datas = pd.DataFrame({'user_ids':user_ids, 'item_ids': item_ids, 'ratings': ratings, 'reviews': reviews})
    datas = datas.sample(num)
    return list(datas['user_ids']), list(datas['item_ids']), list(datas['ratings']), list(datas['reviews'])

temp_user_ids = []
temp_item_ids = []
temp_ratings = []
temp_reviews = []

user_ids = []
item_ids = []
ratings = []
reviews = []

user_review_num = {}
item_review_num = {}

with open(config.data_path_file + "-原始数据") as f:
    line = f.readline()
    while line:
        js = json.loads(line)
        line = f.readline()

        # 跳过没有id的user和item
        if str(js['reviewerID']) == 'unknown':
            print('unknown user_id')
        if str(js['asin']) == 'unknown':
            print('unknown item_id')

        user_id = js['reviewerID']
        item_id = js['asin']
        rating = js['overall']
        review = js['reviewText']

        user_ids.append(user_id)
        item_ids.append(item_id)
        ratings.append(rating)
        reviews.append(review)

        num = user_review_num.setdefault(user_id, 0) + 1
        user_review_num[user_id] = num

        num = item_review_num.setdefault(item_id, 0) + 1
        item_review_num[item_id] = num

del_user_ids = [id for id,num in user_review_num.items() if num < 6]
del_item_ids = [id for id,num in item_review_num.items() if num < 6]

print('user num %d, del user %d' % (len(set(user_ids)), len(del_user_ids)))
print('item num %d, del item %d' % (len(set(item_ids)), len(del_item_ids)))

del_idx = []
for i in range(len(ratings)):
    sys.stdout.write('\r%d / %d' % (i+1, len(ratings)))
    if not (user_ids[i] in del_user_ids and item_ids[i] in del_item_ids):
        temp_user_ids.append(user_ids[i])
        temp_item_ids.append(item_ids[i])
        temp_ratings.append(ratings[i])
        temp_reviews.append(reviews[i])

user_ids = temp_user_ids
item_ids = temp_item_ids
ratings = temp_ratings
reviews = temp_reviews

user_ids, item_ids, ratings, reviews = sample(user_ids, item_ids, ratings, reviews, 400000)

with open(config.data_path_file, 'w') as f:
    for i in range(len(ratings)):
        data = {'reviewerID':user_ids[i], 'asin':item_ids[i], 'overall':ratings[i], 'reviewText':reviews[i]}
        f.write(json.dumps(data) + '\n')

print('完成: ' + config.data_path_file)


