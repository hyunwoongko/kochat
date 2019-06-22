# Author : Hyunwoong
# When : 5/6/2019
# Homepage : github.com/gusdnd852
import os

import numpy as np
import tensorflow as tf
from gensim.models import FastText
from konlpy.tag import Okt

from src.intent.configs import IntentConfigs

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


def inference_embed(data):
    mecab = Okt()
    model = FastText.load(configs.fasttext_path + 'model')
    encode_raw = mecab.morphs(data)
    encode_raw = list(map(lambda x: encode_raw[x] if x < len(encode_raw) else '#', range(encode_length)))
    input = np.array(
        list(map(lambda x: model[x] if x in model.wv.index2word else np.zeros(vector_size, dtype=float), encode_raw)))
    return input


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

            h = tf.nn.relu(tf.nn.bias_add(conv, b_conv1), name="relu")
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


def predict(test_data):
    try:
        tf.reset_default_graph()
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        _, x, _, _, _, y, _, _ = create_graph(train=False)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        dir = os.listdir(configs.model_path)
        num_ckpt = 0
        for i in dir:
            try:
                new_one = int(i.split('-')[1].split('.')[0])
                if num_ckpt < new_one:
                    num_ckpt = new_one
            except:
                pass

        saver.restore(sess, configs.model_path + 'check_point-' + str(num_ckpt) + '.ckpt')
        y = sess.run([y], feed_dict={x: np.array([test_data])})
        score = y[0][0][np.argmax(y)]
        if score > configs.fallback_score:
            return format(np.argmax(y))
        else:
            return None
    except Exception as e:
        raise Exception("error on training: {0}".format(e))
    finally:
        sess.close()


def get_intent(text):
    prediction = predict(np.array(inference_embed(text)).flatten())
    if prediction is None:
        return "폴백"
    else:
        for mapping, num in intent_mapping.items():
            if int(prediction) == num:
                return mapping
