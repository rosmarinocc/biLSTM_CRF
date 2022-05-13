# -*- coding:utf-8 -*-
import re
import keras as kr
import pickle
import numpy as np
from parameters import Parameters as para

TypeDict = {'B': 0, 'M': 1, 'E': 2, 'S': 3}


def convert_word_seq(sentence):
    '''句子转换为字序列'''
    sentence = ''.join(sentence.split(' '))  # 去除' '
    return [s for s in sentence]  # 返回字序列


def convert_BMES_seq(sentence):
    '''句子转换为BMES序列'''
    sentence = re.sub('  ', ' ', sentence)  # 处理2个空格
    L = sentence.split(' ')
    BMES_L = []
    for l in L:
        if len(l) == 1:
            BMES_L.append('S')
        elif len(l) == 2:
            BMES_L.append('B')
            BMES_L.append('E')
        else:
            BMES_L.append('B')
            n_M = len(l)-2
            for i in range(n_M):
                BMES_L.append('M')
            BMES_L.append('E')
    return BMES_L


def read_file(filename):
    word, content, label = [], [], []
    with open(filename, 'r', encoding='utf-8-sig') as f:
        for l in f:
            l = l.strip('\n').strip(' ')
            word_seq = convert_word_seq(l)
            BMES_seq = convert_BMES_seq(l)
            word.extend(word_seq)
            content.append(word_seq)
            label.append(BMES_seq)
    return word, content, label


def word_dict(filename):
    '''
    从训练集获取所有字符的字典
    '''
    word, _, _ = read_file(filename)
    word = set(word)

    key_dict = {}
    key_dict['<PAD>'] = 0  # 填充符
    key_dict['<UNK>'] = 1  # 低频词或未在词表中的词(未登录词)

    j = 2
    for w in word:
        key_dict[w] = j
        j += 1
    with open('./data/word2id.pkl', 'wb') as fw:  # 将建立的字典 保存
        pickle.dump(key_dict, fw)
    return key_dict


def sequence2id(filename):
    '''
    将文字与标签,转换为数字
    '''
    content2id, label2id = [], []
    _, content, label = read_file(filename)
    with open('./data/word2id.pkl', 'rb') as fr:
        key_dict = pickle.load(fr)
    for i in label:
        label2id.append([TypeDict[ii] for ii in i])

    for j in content:
        w = []
        for key in j:
            if key not in key_dict:
                key = '<UNK>'
            w.append(key_dict[key])
        content2id.append(w)
    return content2id, label2id


def batch_iter(content, label, batch_size=para.batch_size):
    Len = len(content)
    x = np.array(content)
    y = np.array(label)
    num_batch = int((Len-1) / batch_size) + 1  # 批数
    indices = np.random.permutation(Len)  # [0,Len)数字随机排序
    # 混洗,打乱顺序
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start = i * batch_size
        end = min((i+1) * batch_size, Len)
        yield x_shuffle[start:end], y_shuffle[start:end]


def process(x_batch):
    '''
       计算一个batch中content最大长度,并且处理成等长
     '''
    len_seq = []
    max_len = max(map(lambda x: len(x), x_batch))  # 计算一个batch中最长长度
    for i in x_batch:
        len_seq.append(len(i))  # 原始长度

    x_pad = kr.preprocessing.sequence.pad_sequences(  # 处理成等长
        x_batch, max_len, padding='post', truncating='post')  # 默认填充0 -- key_dict['<PAD>']

    return x_pad, len_seq
