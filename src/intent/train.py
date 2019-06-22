import os

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import FastText
from konlpy.tag import Okt

from src.intent.configs import IntentConfigs
from src.util.tokenizer import tokenize

configs = IntentConfigs()
# 파라미터 세팅
data = configs.data
encode_length = configs.encode_length
label_size = configs.label_size
filter_sizes = configs.filter_sizes
num_filters = configs.num_filters
intent_mapping = configs.intent_mapping
learning_step = configs.learning_step
learning_rate = configs.learning_rate
vector_size = configs.vector_size


def preprocess_data(tokenizing):
    data['intent'] = data['intent'].map(intent_mapping)

    if tokenizing:
        count = 0
        for i in data['question']:
            data.replace(i, tokenize(i), regex=True, inplace=True)
            if count % 100 == 0:
                print("CURRENT COLLECT : ", count)
            count += 1

    encode = []
    decode = []
    for q, i in data.values:
        encode.append(q)
        decode.append(i)

    return {'encode': encode, 'decode': decode}


def train_vector_model(datas, train):
    path = configs.fasttext_path
    if train:
        mecab = Okt()
        str_buf = datas['encode']
        joinString = ' '.join(str_buf)
        pos1 = mecab.pos(joinString)
        pos2 = ' '.join(list(map(lambda x: '\n' if x[1] in ['Punctuation'] else x[0], pos1))).split('\n')
        morphs = list(map(lambda x: mecab.morphs(x), pos2))
        print("BUILD MODEL")
        model = FastText(size=vector_size,
                         window=3,
                         workers=8,
                         min_count=2,
                         sg=1,
                         iter=1500)
        model.build_vocab(morphs)
        print("BUILD COMPLETE")

        print("TRAIN START")
        model.train(morphs, total_examples=model.corpus_count,
                    epochs=model.epochs,
                    compute_loss=True)

        if not os.path.exists(path):
            os.makedirs(path)

        model.save(path + 'model')
        print("TRAIN COMPLETE")
        return model
    else:
        print("LOAD SAVED MODEL")
        return FastText.load(path + 'model')


train_data_list = preprocess_data(tokenizing=configs.tokenizing)
model = train_vector_model(train_data_list, train=configs.train_fasttext)


########################################
########################################
########################################

def load_csv(data_path):
    df_csv_read = pd.DataFrame(data_path)
    return df_csv_read


def embed(data):
    mecab = Okt()
    inputs = []
    labels = []
    for encode_raw in data['encode']:
        encode_raw = mecab.morphs(encode_raw)
        encode_raw = list(map(lambda x: encode_raw[x] if x < len(encode_raw) else '#', range(encode_length)))
        input = np.array(list(
            map(lambda x: model[x] if x in model.wv.index2word else np.zeros(vector_size, dtype=float),
                encode_raw)))
        inputs.append(input.flatten())

    for decode_raw in data['decode']:
        label = np.zeros(label_size, dtype=float)
        np.put(label, decode_raw, 1)
        labels.append(label)
    return inputs, labels


def inference_embed(data):
    mecab = Okt()
    encode_raw = mecab.morphs(data)
    encode_raw = list(map(lambda x: encode_raw[x] if x < len(encode_raw) else '#', range(encode_length)))
    input = np.array(
        list(map(lambda x: model[x] if x in model.wv.index2word else np.zeros(vector_size, dtype=float), encode_raw)))
    return input


def get_test_data():
    train_data, train_label = embed(load_csv(train_data_list))
    test_data, test_label = embed(load_csv(train_data_list))
    return train_label, test_label, train_data, test_data


def create_graph(train=True):
    x = tf.placeholder("float", shape=[None, encode_length * vector_size], name='x')
    y_target = tf.placeholder("float", shape=[None, label_size], name='y_target')
    x_image = tf.reshape(x, [-1, encode_length, vector_size, 1], name="x_image")
    l2_loss = tf.constant(0.0)
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, vector_size, 1, num_filters]
            W_conv1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

            conv = tf.nn.conv2d(
                x_image,
                W_conv1,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")

            h = tf.nn.leaky_relu(tf.nn.bias_add(conv, b_conv1), name="relu")
            pooled = tf.nn.max_pool(h,
                                    ksize=[1, encode_length - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name="pool")
            pooled_outputs.append(pooled)

    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    keep_prob = 1.0
    if train:
        keep_prob = tf.placeholder("float", name="keep_prob")
        h_pool_flat = tf.nn.dropout(h_pool_flat, keep_prob)

    W_fc1 = tf.get_variable(
        "W_fc1",
        shape=[num_filters_total, label_size],
        initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[label_size]), name="b")
    l2_loss += tf.nn.l2_loss(W_fc1)
    l2_loss += tf.nn.l2_loss(b_fc1)
    y = tf.nn.xw_plus_b(h_pool_flat, W_fc1, b_fc1, name="scores")
    predictions = tf.argmax(y, 1, name="predictions")
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_target)
    cross_entropy = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_predictions = tf.equal(predictions, tf.argmax(y_target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    return accuracy, x, y_target, keep_prob, train_step, y, cross_entropy, W_conv1


def train_intent():
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        labels_train, labels_test, data_filter_train, data_filter_test = get_test_data()
        accuracy, x, y_target, keep_prob, train_step, y, cross_entropy, W_conv1 = create_graph(train=True)
        path = configs.model_path
        if not os.path.exists(path):
            os.makedirs(path)
        num_ckpt = 0
        if len(os.listdir(path)) > 3:
            print('LOAD CNN MODEL')
            dir = os.listdir(path)
            for i in dir:
                try:
                    new_one = int(i.split('-')[1].split('.')[0])
                    if num_ckpt < new_one:
                        num_ckpt = new_one
                except:
                    pass
            restorer = tf.train.Saver(tf.all_variables())
            restorer.restore(sess, configs.model_path + 'check_point-' + str(num_ckpt) + '.ckpt')
        else:
            sess.run(tf.global_variables_initializer())

        for i in range(learning_step):
            sess.run(train_step, feed_dict={x: data_filter_train, y_target: labels_train, keep_prob: 0.5})
            index = i + num_ckpt
            if i % 100 == 0:
                train_accuracy = sess.run(accuracy,
                                          feed_dict={x: data_filter_train, y_target: labels_train, keep_prob: 1})
                print("step %d, training accuracy: %.3f" % (index, train_accuracy))
            if i != 0 and i % 100 == 0:
                saver = tf.train.Saver(tf.all_variables())
                saver.save(sess, path + 'check_point-' + str(index) + '.ckpt')
                print("Save Models checkpoint : %d" % index)
