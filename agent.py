import os
import sys
import psutil
import gc
# import objgraph
# import guppy
# from pympler import tracker, muppy, summary
from memory_profiler import profile


import numpy as np
import random

from collections import deque

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import Callback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 1 to filter logs, 2 warnings, 3 for errors
# tr = tracker.SummaryTracker()
# hp = guppy.hpy()


# class MemoryUsageCallback(Callback):
#   '''Monitor memory usage on epoch begin and end.'''
#
#   def on_epoch_begin(self,epoch,logs=None):
#     print('**Epoch {}**'.format(epoch))
#     print('Memory usage on epoch begin: {}'.format(psutil.Process(os.getpid()).memory_info().rss))
#
#   def on_epoch_end(self,epoch,logs=None):
#     print('Memory usage on epoch end:   {}'.format(psutil.Process(os.getpid()).memory_info().rss))
#     gc.collect()


class DQN:
    def __init__(self, input_dim, n_actions, model_path=None):
        # Initialize attributes
        self._input_dim = input_dim
        self._action_size = n_actions
        self.model_path = model_path
        self.learn_step = 0
        self.update_interval = 50
        self.batch_size = 32
        self.memory_size = 50000

        # Initialize discount and exploration rate
        self.gamma = 0.65
        self.learning_rate = 0.001

        # Initialize epsilon parameters
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.epsilon_decay = -0.1

        # Is training, initialize memory and build model
        if not self.model_path:
            self.experience_replay = deque(maxlen=self.memory_size)
            self.q_network = self._build_compile_model()
            self.target_network = self._build_compile_model()
            self.align_target_model()

        # Is testing, load model
        elif self.model_path:
            self.q_network = self.load_model()

    def _build_compile_model(self):
        model = Sequential()

        # Convolutional layers
        model.add(Conv2D(32, (2, 4), strides=(1, 2), activation='relu', input_shape=self._input_dim))
        model.add(Conv2D(32, (2, 4), strides=(1, 2), activation='relu'))
        model.add(Conv2D(32, (2, 2), strides=(1, 3), activation='relu'))

        # Flatten layer
        model.add(Flatten())

        # Fully connected layers
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def save_model(self, model_path):
        print('Saving model')
        save_model(self.q_network, filepath=os.path.join(model_path, 'training_model.h5'))

    def load_model(self):
        model_path = os.path.join(self.model_path, 'training_model.h5')
        if os.path.isfile(model_path):
            print('Model found')
            return load_model(filepath=model_path)
        else:
            sys.exit('Model not found')

    def store(self, state, action, reward, next_state):
        self.experience_replay.append((state, action, reward, next_state))
        # print('experience_replay size:', len(self.experience_replay))

    # @profile
    def align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def predict(self, state_input, target=False, ):
        if not target:
            return self.q_network.predict(state_input, verbose=0)
        if target:
            return self.target_network.predict(state_input, verbose=0)

    def fit(self, x, y):
        return self.q_network.fit(x, y, epochs=1, verbose=0)

    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            # print('---random draw---')
            return random.randint(0, self._action_size - 1)

        q_value = self.predict(np.reshape(state, [1, 60, 16, 2]))
        return np.argmax(q_value[0])

    def get_epsilon(self, episode):
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(self.epsilon_decay * episode)

    # @profile
    def train(self):
        # print(f'------memory total {self.learn_step}------')
        # all_obj = muppy.get_objects()
        # sum1 = summary.summarize(all_obj)
        # summary.print_(sum1)
        #
        # print(f'---memory difference--- {self.learn_step}')
        # tr.print_diff()

        # callbacks = [MemoryUsageCallback()]

        # Check to replace target network
        if self.learn_step % self.update_interval == 0:
            self.align_target_model()
            # print(self.learn_step, 'update target network')

        # Sample batch memory from all experiences
        # print('memory size:', len(self.experience_replay), sys.getsizeof(self.experience_replay))
        if len(self.experience_replay) > self.batch_size:
            minibatch = random.sample(self.experience_replay, self.batch_size)
        else:
            minibatch = self.experience_replay

        states = np.array([val[0] for val in minibatch])
        next_states = np.array([val[3] for val in minibatch])

        current_qs = self.predict(states, target=False)
        t = self.predict(next_states, target=True)

        x = np.zeros((self.batch_size, 60, 16, 2))
        y = np.zeros((self.batch_size, self._action_size))

        # print('before', sys.getsizeof(x), sys.getsizeof(y))

        for i, value in enumerate(minibatch):
            state, action, reward, next_state = value
            current_q = current_qs[i]  # get the current Q
            current_q[action] = reward + self.gamma * np.amax(t[i])  # update Q

            x[i] = state
            y[i] = current_q

        self.q_network.fit(x, y, epochs=1, verbose=0)
        # print(sys.getrefcount(states))
        # print(sys.getrefcount(t))
        # print('after', sys.getsizeof(x), sys.getsizeof(y))

        # A workaround to the memory leak problem of model.fit/predict in tensorflow
        tf.keras.backend.clear_session()
        # gc.collect()

        self.learn_step += 1
        # print(self.learn_step)
