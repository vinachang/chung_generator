import time
import numpy as np
import pickle
import tensorflow as tf

from config import model_path, words_path, word_to_id_path, training_data_path


def build_rnn_graph(inputs, hidden_size, is_training, keep_prob, num_layers,
                    batch_size, num_steps):
    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, reuse=not is_training)
    if is_training and keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=keep_prob)

    cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)])

    initial_state = cell.zero_state(batch_size, tf.float32)
    state = initial_state

    inputs = tf.unstack(inputs, num=num_steps, axis=1)
    outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
                               initial_state=initial_state)
    output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
    return initial_state, output, state

def lstm_model(max_grad_norm = 5, num_layers=2, num_steps = 20,
              hidden_size = 200, keep_prob = 1.0, batch_size = 20,
              vocab_size = 3000, optimization='gd', is_training=True):
    embedding = tf.get_variable(
        "embedding", [vocab_size, hidden_size], dtype=tf.float32)
    user_inputs = tf.placeholder(tf.int32, shape=(batch_size, num_steps))
    targets = tf.placeholder(tf.int32, shape=(batch_size, num_steps))
    inputs = tf.nn.embedding_lookup(embedding, user_inputs)

    if keep_prob <= 1:
        inputs = tf.nn.dropout(inputs, keep_prob)

    initial_state, output, final_state = build_rnn_graph(inputs, hidden_size,
                                                         is_training, keep_prob,
                                                         num_layers, batch_size,
                                                         num_steps)
    softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size],
                                dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

    # Reshape logits to be a 3-D tensor for sequence loss
    logits_reshaped = tf.reshape(logits, [batch_size, num_steps, vocab_size])
    loss = tf.contrib.seq2seq.sequence_loss(logits_reshaped, targets,
                                            tf.ones([batch_size, num_steps],
                                                    dtype=tf.float32),
                                            average_across_timesteps=False,
                                            average_across_batch=True)
    cost = tf.reduce_sum(loss)

    tensors = {
        "user_inputs":	user_inputs,
        "targets":	targets,
        "initial_state":	initial_state,
        "final_state":	final_state,
        "output": 	output,
        "logits":	logits,
        "num_steps": num_steps,
        "cost":	cost,
    }

    if not is_training:
        return tensors

    lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      max_grad_norm)
    if optimization == 'momentum':
        optimizer = tf.train.MomentumOptimizer(lr, 0.9)
    elif optimization == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(lr)
    else:
        optimizer = tf.train.GradientDescentOptimizer(lr)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

    new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    lr_update = tf.assign(lr, new_lr)

    return {
        "lr_update": lr_update,
        "lr": lr,
        "new_lr":	new_lr,
        "train_op":	train_op,
        **tensors
    }

def run_epoch(data, session, model, train_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    num_steps = model['num_steps']
    num_batches = (data.shape[1] - 1) // num_steps

    # initialize RNN cell states to be all zero
    state = session.run(model['initial_state'])

    fetches = {
        "cost": model['cost'],
        "final_state": model['final_state'],
    }

    # train model
    if train_op is not None:
        fetches["train_op"] = train_op

    for batch in range(num_batches):
        feed_dict = {
            model['user_inputs']: data[:, batch * num_steps: (batch + 1) * num_steps],
            model['targets']: data[:, batch * num_steps + 1:  (batch + 1) * num_steps + 1],
        }
        for i, (c, h) in enumerate(model['initial_state']):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]
        costs += cost

        if verbose and batch % (num_batches // 10) == 10:
            iters = num_steps * (batch + 1)
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (batch * 1.0 / num_batches, np.exp(costs / iters),
                   iters * data.shape[0] * 1 /
                   (time.time() - start_time)))

    return np.exp(costs / (data.shape[1] - 1))


def train_word_generator(training, validation, words, word_to_id,
                         init_scale=0.1, learning_rate=1.0, decay_after=4,
                         learning_rate_decay=0.5, keep_prob=0.95,
                         num_epochs=10, batch_size=20,
                         num_layers=2, num_steps = 20,
                         hidden_size=200, vocab_size = 3000, max_grad_norm = 5,
                         optimization='gd',
                         name='model'):
    """
    Train a word generator with LSTM.

    Parameter
    ------
    training: 1D numpy array
        training data with ids represent different words
    validation: 1D numpy array
    words: list
        words list used to convert generated ids back to words
    word_to_id: dict
        a dict mapping word to id, used to convert user input to id for word
        generation
    init_scale: float, optional
        initializer range
    num_layers: int, optional
        number of LSTM cells
    hidden_size: int, optional
        number of units in a LSTM cell
    learning_rate: float, optional
    learning_rate_decay: float, optional
        the rate at which learning rate decays
    decay_after: int, optional
        number of epochs after which to start learning rate decay
    keep_prob: float, optional
        percentage of neurons to keep. if keep_prob is 1 then no dropout is applied.
    num_epochs: int, optional
    batch_size: int, optional
    num_steps: int, optional
    vocab_size: int, optional
    max_grad_norm: int, optional
    optimization: one of ['gd', 'momentum', 'rmsprop']
    name: str, optional
        name of the model (used in tensorflow summary and model saving)
    """
    # reshape training and validation according to batch_size
    training = np.array(training[:(
        len(training) // batch_size) * batch_size]).reshape(batch_size, -1)
    validation = np.array(validation).reshape(1, -1)

    # save words and word_to_id to be used in prediction
    with open(words_path, 'wb') as f:
        pickle.dump(words, f)
    with open(word_to_id_path, 'wb') as f:
        pickle.dump(word_to_id, f)

    param = {
        'max_grad_norm': max_grad_norm,
        'num_layers': num_layers,
        'hidden_size': hidden_size,
        'vocab_size': vocab_size,
    }

    # initialize graph
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)

    with tf.name_scope("train"):
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = lstm_model(is_training=True, num_steps=num_steps,
                          optimization=optimization,
                          batch_size=batch_size, keep_prob=keep_prob, **param)

    with tf.name_scope("valid"):
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = lstm_model(
                batch_size=1, num_steps=1, is_training=False, **param)

    train_writer = tf.summary.FileWriter('./summary/%s/train' % name)
    valid_writer = tf.summary.FileWriter('./summary/%s/valid' % name)

    saver = tf.train.Saver()

    # use just one thread
    session_conf = tf.ConfigProto(
          intra_op_parallelism_threads=1,
          inter_op_parallelism_threads=1)
    with tf.Session(config=session_conf) as session:
        session.run(tf.global_variables_initializer())
        for i in range(num_epochs):

            # update learning rate
            lr_decay = learning_rate_decay ** max(i +
                                                  1 - decay_after, 0.0)
            session.run(m['lr_update'], feed_dict={
                        m['new_lr']: learning_rate * lr_decay})
            new_lr = session.run(m['lr'])
            print("Epoch: %d Learning rate: %.3f" %
                  (i + 1, new_lr))

            train_perplexity = run_epoch(training, session, m, train_op=m['train_op'],
                                         verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" %
                  (i + 1, train_perplexity))
            epoch_perplexity_train = tf.Summary(value=[
              tf.Summary.Value(tag="perplexity", simple_value=train_perplexity),
            ])
            lr_summary = tf.Summary(value=[
              tf.Summary.Value(tag="learning_rate", simple_value=new_lr),
            ])
            train_writer.add_summary(epoch_perplexity_train, i + 1)
            train_writer.add_summary(lr_summary, i + 1)
            train_writer.flush()

            valid_perplexity = run_epoch(validation, session, mvalid)
            print("Epoch: %d Valid Perplexity: %.3f" %
                  (i + 1, valid_perplexity))
            epoch_perplexity_valid = tf.Summary(value=[
              tf.Summary.Value(tag="perplexity", simple_value=valid_perplexity),
            ])
            valid_writer.add_summary(epoch_perplexity_valid, i + 1)
            valid_writer.flush()

            if i % 5 == 0:
                # save session every 5 epochs
                saver.save(session, "%s/%s/model" % (model_path, name), i)

        saver.save(session, "%s/%s/model" % (model_path, name))
