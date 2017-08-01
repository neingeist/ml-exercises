from random import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.layers import TimeDistributed


def get_sequence1(n_timesteps):
    # create a sequence of random numbers in [0,1]
    X = np.array([random() for _ in range(n_timesteps)])
    # calculate cut-off value to change class values
    limit = n_timesteps/4.0
    # determine the class outcome for each item in cumulative sequence
    y = np.array([0 if x < limit else 1 for x in np.cumsum(X)])
    return X, y


def get_sequence2(n_timesteps):
    """generate random numbers, y is 1 for steps > 0.5"""
    X = np.array([random() for _ in range(n_timesteps)])

    y = X-np.concatenate(([0], X))[:-1] > 0.5
    y = y.astype('float')
    return X, y


def get_sequence3(n_timesteps):
    """generate a sequence with small and big steps, the net should learn to
       detect the big steps"""
    small_step = random()
    big_step = small_step + 2*random()
    big_step_prob = 0.1

    X = np.zeros(n_timesteps)
    X[0] = random()
    for i in range(1, n_timesteps):
        step = big_step if random() < big_step_prob else small_step
        X[i] = X[i-1] + step
    X = X.astype('float32')

    y = np.logical_not(
            np.isclose((X-np.concatenate(([0], X))[:-1]), small_step)
        )
    y = y.astype('float32')
    return X, y


get_sequence = get_sequence3


def get_sequences(n_sequences, n_timesteps):
    s = [get_sequence(n_timesteps) for i in range(n_sequences)]
    Xs = np.array([X for X, _ in s])
    ys = np.array([y for _, y in s])
    return Xs, ys


model = Sequential()
rnn = LSTM  # or GRU
model.add(rnn(128, input_shape=(10, 1), return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

n_samples = 10000
Xs, ys = get_sequences(n_samples, 10)
Xs = np.reshape(Xs, (n_samples, 10, 1))
ys = np.reshape(ys, (n_samples, 10, 1))
print(Xs[0], ys[0])


model.fit(Xs, ys,
          validation_split=0.2,
          batch_size=32,
          epochs=40,
          shuffle=True)
