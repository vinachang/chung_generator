import json
import random
from collections import Counter
import re
import os
import pickle

from config import model_path, words_path, word_to_id_path, training_data_path


multi_newlines = '[\n\r]{3,}'
replacements = {
    '（': '(',
    '）': ')',
    ',': '，',
    '?': '？',
    ':': '：',
    '!': '！',
    '％': '%',
    '-': '─',
    '—': '─',
    '\u202a': '',
    '\u200b': '',
    '\xad': '',
    '\xa0': '',
    '\u3000': ' ',
    '’': "'",
    '”': '"',
    '“': '"',
    '～': '~',
    '‧': '',
    '·': '',
    '＠': '@',
    '…': '...',
    '&amp;': '&',
    '&gt;': '>',
}


def process(text):
    "Remove extra space, invalid chars and convert punctuations."
    for key in replacements:
        text = text.replace(key, replacements[key])
    text = re.sub(multi_newlines, '\n\n', text.strip())
    return text


def get_vocabulary(training):
    counter = Counter(training)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    vocabulary_to_id = dict(zip(words, range(len(words))))
    return vocabulary_to_id, words


def get_data():
    # check if data have been processed before; if so, read from file
    if (os.path.isfile(words_path) and os.path.isfile(word_to_id_path) and
            os.path.isfile(training_data_path)):
        with open(words_path, 'rb') as f:
            words = pickle.load(f)
        with open(word_to_id_path, 'rb') as f:
            word_to_id = pickle.load(f)
        with open(training_data_path, 'rb') as f:
            training, validation = pickle.load(f)
    else:
        with open('crawler/articles.json', 'r') as f:
            articles = json.load(f)

        # concat article title and article content
        full_texts = ['<t>{}</t>{}'.format(article['title'],
                                           process(article['content']))
                      for article in articles]

        # split into training and validation set
        random.shuffle(full_texts)
        training, validation = ['\n'.join(texts) for texts in [full_texts[26:],
                                                               full_texts[:26]]]

        word_to_id, words = get_vocabulary(training)

        # convert word to id
        training = [word_to_id[v] for v in training]
        validation = [word_to_id[v] for v in validation if v in word_to_id]

        with open(words_path, 'wb') as f:
            pickle.dump(words, f)
        with open(word_to_id_path, 'wb') as f:
            pickle.dump(word_to_id, f)
        with open(training_data_path, 'wb') as f:
            pickle.dump((training, validation), f)

    return word_to_id, words, training, validation
