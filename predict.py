# -*- coding:utf-8 -*-
import pickle
from data_process import convert_word_seq, process, batch_iter, sequence2id, read_file
from parameters import Parameters as para
from biLSTM_CRF import biLstm_crf
import tensorflow as tf
import numpy as np


def cutting(sentence, label_line):
    word_cut = ''
    wordlist = convert_word_seq(sentence)
    # TypeDict = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
    for i in range(len(label_line)):
        if label_line[i] == 2:
            word_cut += wordlist[i]
            word_cut += ' '
        elif label_line[i] == 3:
            word_cut += ' '
            word_cut += wordlist[i]
            word_cut += ' '
        else:
            word_cut += wordlist[i]
    return word_cut


def val():
    label = []
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    latest_save_path = tf.train.latest_checkpoint('./checkpoints')
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=latest_save_path)

    content, _ = sequence2id(para.eva)
    pre_label = model.predict(session, content)
    label.extend(pre_label)

    return label


if __name__ == '__main__':
    para = para

    model = biLstm_crf()

    label = val()
    with open(para.eva, 'r', encoding='utf-8-sig') as f:
        sentences = [line.strip('\n') for line in f]

    for i in range(len(sentences)):
        sentence_cut = cutting(sentences[i], label[i])
        print(sentences[i])
        print(sentence_cut)
