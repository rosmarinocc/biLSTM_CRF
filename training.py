import os
from parameters import Parameters as para
from biLSTM_CRF import biLstm_crf
import tensorflow as tf
from data_process import sequence2id, process, batch_iter

def train():
    tensorboard_dir = './tensorboard'
    save_dir = './checkpoints'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')


    tf.summary.scalar('loss', model.loss)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    content_train, label_train = sequence2id(para.train)
    content_test, label_test = sequence2id(para.test)

    for epoch in range(para.epochs):
        print('Epoch:', epoch+1)
        num_batchs = int((len(content_train) - 1) / para.batch_size) + 1 # 轮数
        batch_train = batch_iter(content_train, label_train)
        for x_batch, y_batch in batch_train:
            x_batch, seq_leng_x = process(x_batch)
            y_batch, seq_leng_y = process(y_batch)
            feed_dict = model.feed_data(x_batch, y_batch, seq_leng_x, para.keep_pro)
            _, global_step, loss, tain_summary = session.run([model.optimizer, model.global_step, model.loss, merged_summary],
                                                             feed_dict=feed_dict)
            if global_step % 100 == 0:
                test_loss = model.test(session, content_test, label_test)
                print('global_step:', global_step, 'train_loss:', loss, 'test_loss:', test_loss)

            if global_step % (2*num_batchs) == 0:# 两个epoch保存一个
                print('Saving Model...')
                saver.save(session, save_path=save_path, global_step=global_step)
        para.learning_rate *= para.lr


if __name__ == '__main__':
    para = para
    model = biLstm_crf()
    train()