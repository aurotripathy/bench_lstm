"""
Attributions:
code loosely based on https://github.com/stefbraun/rnn_benchmarks
Arxiv paper https://arxiv.org/abs/1806.01818
Description:
Benchmarking a single-layer LSTM with:
- 320 hidden units
- 100 time steps
- batch size 64
- 1D input features, size 125
- output 10 classes
"""

import time as timer

# from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.layers import Input, CuDNNLSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from utils import get_batch, set_hyperparams, print_results


def validate_lstm_in_out(model):
    """ Detect any dimension mismatch issues"""
    assert (model.layers[-1].input_shape == (None, hidden_units_size))
    # final projection output size (rnn_size, classes)
    assert (model.layers[-1].get_weights()
	    [0].shape == (hidden_units_size, classes))
    output = model.predict(bX)
    assert (output.shape == (batch_size, classes))


def train_lstm():

    # Create symbolic vars
    x = Input(shape=(None, in_dim), dtype='float32', name='input')

    # Create network
    # fw_cell = LSTM(hidden_units_size, return_sequences=False,
    #                implementation=2)(x)
    fw_cell = CuDNNLSTM(hidden_units_size, return_sequences=False)(x)

    h3 = Dense(classes, activation='softmax', use_bias=False)(fw_cell)
    model = Model(inputs=x, outputs=h3)
    validate_lstm_in_out(model)
    start = timer.perf_counter()
    model.compile(optimizer='Adam', loss='categorical_crossentropy')
    end = timer.perf_counter()
    print('>>> Model compilation took {:.1f} seconds'.format(end - start))

    # Print parameter count
    params = model.count_params()
    print('# network parameters: ' + str(params))

    # Start training
    batch_time = []
    batch_loss = []
    train_start = timer.perf_counter()
    for i in range(nb_batches):
        batch_start = timer.perf_counter()
        loss = model.train_on_batch(
            x=bX, y=to_categorical(bY, num_classes=classes))
        batch_end = timer.perf_counter()
        batch_time.append(batch_end - batch_start)
        batch_loss.append(loss)
        train_end = timer.perf_counter()
    print_results(batch_loss, batch_time, train_start, train_end)


if __name__ == '__main__':

    hidden_units_size, learning_rate, seq_len, batch_size, nb_batches = set_hyperparams()
    classes = 10
    in_dim = 125
    bX, bY = get_batch(
	shape=(batch_size, seq_len, in_dim), classes=classes)
    print("Hidden units:{}, Learning Rate:{}, LSTM time steps:{}, Batch size:{}, Batches:{}".format(hidden_units_size,
			                                                                            learning_rate, seq_len,
												    batch_size, nb_batches))

    train_lstm()
    
