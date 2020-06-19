"""
Created on 2019/3/2 15:58

@author: zhouweixin
@note: 数据配置文件
"""

import os
import time

# 切换数据类型
data_type = 'toy_game'

assert data_type in ['kindle', 'movie', 'instrument', 'toy_game', 'music', 'yelp']

toy_game_file = 'reviews_Toys_and_Games_5.json'
instrument_file = 'reviews_Musical_Instruments_5.json'
kindle_file = 'reviews_Kindle_Store_5.json'
music_file = 'Digital_Music_5.json'
movie_file = 'reviews_Movies_and_TV_5.json'
yelp_file = 'yelp.json'

data_type2data_file = {
    'music': music_file,
    'movie': movie_file,
    'toy_game': toy_game_file,
    'kindle': kindle_file,
    'instrument': instrument_file,
    'yelp': yelp_file,
}

# 数据根路径
data_root = '../../narre_all_data'
# 数据源文件
data_file = data_type2data_file[data_type]
# 数据全路径
data_path_file = os.path.join(data_root, data_type, data_file)
# 输出路径
output_path = os.path.join(data_root, data_type)

# GloVe源文件
glove_file = 'glove.6B.300d.txt'
glove_file = 'glove.840B.300d.txt'
glove_file = 'GoogleNews-vectors-negative300.bin'

stopwords= os.path.join(data_root, 'stopwords.txt')

# GloVe全路径
glove_path_file = os.path.join(data_root, 'glove', glove_file)
# GloVe init全路径
glove_init_file = os.path.join(data_root, data_type, 'glove.init.npy')

# 数据集比例
data_ratio = 1
train_val_test_ratio = [0.8, 0.1, 0.1]

# dictionary输出文件
dictionary_file = 'dictionary.pkl'
dictionary_path_file = os.path.join(data_root, data_type, dictionary_file)

# 评论数量比例
review_num_ratio = 0.9
# 评论长度比例
review_len_ratio = 0.9

# 实验结果输出文件
result_output_path = os.path.join(output_path, 'result', time.strftime('%Y%m%d-%H%M%S', time.localtime()))

