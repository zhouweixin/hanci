import json
import os


yelp_path = '../../narre_all_data/yelp/yelp_dataset'


def load_user2num():
    user2num = {}
    with open(os.path.join(yelp_path, 'user.json'), 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break

            user = json.loads(line)
            id = user['user_id']
            num = user['review_count']

            user2num[id] = num

    user2num = dict(sorted(user2num.items(), key=lambda item:item[1]))
    return user2num

user2num = load_user2num()
print(user2num)