training_param = {
    'init_scale': 0.1,
    'learning_rate': 1.0,
    'learning_rate_decay': 0.5,
    'decay_after': 10,
    'num_epochs': 20,
    'batch_size': 20,
    'keep_prob': 0.82,
    'max_grad_norm': 5,
    'num_layers': 1,
    'num_steps': 38,
    'hidden_size': 270,
}

model_path = './model'
words_path = './data/vocabulary.pkl'
word_to_id_path = './data/vocabulary_to_id.pkl'
training_data_path = './data/training_data.pkl'
