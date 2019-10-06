"""
Created on 2019/3/2 15:58

@author: zhouweixin
@note: 主函数
"""

from util.utils import load_review_att
from core.dataset import Dictionary


dictionary = Dictionary.load_from_file()
load_review_att(dictionary.idx2word)