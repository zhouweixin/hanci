"""
Created on 2019/3/2 15:58

@author: zhouweixin
@note: 创建字典和筛选预训练词向量
"""
import json
import numpy as np
from core import config
from core.dataset import Dictionary


def create_dictionary(data_path_file=config.data_path_file):
    """
    创建字典： word2idx, idx2word
    """

    dictionary = Dictionary()

    # 加载所有review
    print("loading all reviews...")
    reviews = []
    with open(data_path_file) as f:
        line = f.readline()
        while line:
            js = json.loads(line)
            line = f.readline()

            # 跳过没有user_id和item_id的数据
            if str(js['reviewerID']) == 'unknown':
                print('unknown user_id')
                continue

            if str(js['asin']) == 'unknown':
                print('unknown item_id')
                continue

            reviews.append(js['reviewText'])

    print('reviews.len = %d' % len(reviews))

    for review in reviews:
        dictionary.tokenize(review, True)

    return dictionary


def create_glove_embedding_init(idx2word, glove_path_file=config.glove_path_file):
    """
    从GloVec中挑选词向量
    :param idx2word:
    :param glove_path_file:
    """
    if 'GoogleNews' in glove_path_file:
        return embedding_from_google(idx2word, glove_path_file)

    return embedding_from_glove(idx2word, glove_path_file)


def embedding_from_glove(idx2word, glove_path_file=config.glove_path_file):
    """
        从GloVec中挑选词向量
        :param idx2word:
        :param glove_path_file:
        :return: weights需要的词向量, word2emb所有词向量
        """
    word2emb = {}

    weights = np.zeros([len(idx2word), 300], dtype=np.float32)

    # 加载所有词向量
    with open(glove_path_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        values = line.split(" ")

        # 第一个是单词, 后面是词向量
        word = values[0]
        values = values[1:]
        word2emb[word] = np.array(values, dtype=np.float32)

    # 挑选需要的词向量
    for idx, word in enumerate(idx2word):

        # 找不到的词跳过
        if word not in word2emb:
            continue

        weights[idx] = word2emb[word]

    return weights, word2emb


def embedding_from_google(idx2word, glove_path_file=config.glove_path_file):
    """
        从GloVec中挑选词向量
        :param idx2word:
        :param glove_path_file:
        :return: weights需要的词向量, word2emb所有词向量
        """
    weights = np.zeros([len(idx2word), 300], dtype=np.float32)

    word2emb = {}
    with open(glove_path_file, 'rb') as f:
        header = f.readline()
        vocab_size, embedding_size = map(int, header.split())

        binary_len = np.dtype('float32').itemsize * embedding_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                ch = ch.decode(encoding='unicode_escape')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)

            embedding = np.fromstring(f.read(binary_len), dtype='float32')
            assert len(embedding) == 300
            word2emb[word] = embedding

    # 挑选需要的词向量
    for idx, word in enumerate(idx2word):

        # 找不到的词跳过
        if word not in word2emb:
            continue

        weights[idx] = word2emb[word]

    return weights, word2emb


dictionary = create_dictionary()
dictionary.dump_to_file(config.dictionary_path_file)

dictionary = dictionary.load_from_file(config.dictionary_path_file)
dictionary.process_review()

weights, word2emb = create_glove_embedding_init(dictionary.idx2word)
np.save(config.glove_init_file, weights)
print("saved %s" % config.glove_init_file)
print("完成")
