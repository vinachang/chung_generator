from data import get_data
from model import train_word_generator
from config import training_param

def main():
    word_to_id, words, training, validation = get_data()
    training_param['vocab_size'] = len(words)
    train_word_generator(training, validation, words, word_to_id,
                   **training_param)

if __name__ == '__main__':
    main()
