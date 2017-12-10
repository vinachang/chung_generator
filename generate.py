import os.path
import argparse

import tensorflow as tf
import pickle
import numpy as np

from data import get_data
from config import model_path, words_path, word_to_id_path, training_param
from model import lstm_model

parser = argparse.ArgumentParser(description='Generate text from pre-trained LSTM model')
parser.add_argument('-l', '--length', default=100, type=int,
                    help='number of words to generate')


def softmax(logits, len_words):
    logits = logits.flatten()[:len_words]
    exp = np.exp(logits)
    return exp / np.sum(exp)

def generate_one_word(logits, words):
    # softmax
    distribution = softmax(logits, len(words))
    # sample from distribution
    index = np.random.choice(range(distribution.shape[0]), size=1, p=distribution)[0]

    return index, words[index]

def generate(session, model, inputs, word_to_id, words, size=20):
    """
    inputs: string
    """
    # convert input to np array
    text = [s for s in inputs]
    inputs = np.array([word_to_id[s] for s in inputs if s in word_to_id]).reshape(1, -1)

    # initialize state
    state = session.run(model['initial_state'])
    fetches = {
        "final_state": model['final_state'],
        "logits": model['logits']
    }

    # feed in all user inputs
    for w in range(inputs.shape[1]):
        feed_dict = {
            model['user_inputs']: inputs[:, w].reshape(1, 1),
        }
        for i, (c, h) in enumerate(model['initial_state']):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        state = vals["final_state"]
    logits = vals['logits']

    index = int(inputs[:, -1])
    for w in range(size):
        feed_dict = {
            model['user_inputs']: np.array([[index]]),
        }
        for i, (c, h) in enumerate(model['initial_state']):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        state = vals["final_state"]
        logits = vals['logits']
        index, word = generate_one_word(logits, words)
        text.append(word)
    return text

def main():
    model_name = '270_0.8201_1_38_10_gd'
    model_path = './model/%s/model' % model_name

    if not os.path.isfile('%s.index' % model_path):
        print("Please run python train.py to train the model first.")
        exit(1)
    args = parser.parse_args()
    length = args.length

    print("Restoring model...")
    with open(words_path, 'rb') as f:
        words = pickle.load(f)
    with open(word_to_id_path, 'rb') as f:
        word_to_id = pickle.load(f)

    param = {
        'vocab_size': len(words),
        'num_layers': training_param['num_layers'],
        'hidden_size': training_param['hidden_size'],
    }
    num_steps = training_param['num_steps']
    batch_size = training_param['batch_size']
    keep_prob = training_param['keep_prob']
    init_scale = training_param['init_scale']

    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    with tf.name_scope("train"):
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = lstm_model(is_training=True, num_steps=int(num_steps),
                           batch_size=batch_size, keep_prob=float(keep_prob),
                           **param)

    with tf.name_scope("valid"):
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = lstm_model(
                batch_size=1, num_steps=1, is_training=False, **param)

    saver = tf.train.Saver()
    session =  tf.Session()
    session.run(tf.global_variables_initializer())
    saver.restore(session, model_path)

    while True:
        user_input = input("Enter any text to collaborate with the generator or 'END' to stop:\n")
        if user_input == 'END':
            exit(0)
        print(''.join(generate(session, mvalid, user_input, word_to_id, words, size=length)))
        print('\n')

if __name__ == '__main__':
    main()
