''' In this code we will define the model for training and testing'''

import config

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import LSTM, Input, Embedding, Dense
from keras.optimizers import Adam

class LanguageModel:

    def train_model(self, input_sequences, output_sequences,
                    embedding_matrix, max_sequence_len, actual_vocab_size):
        '''

        :param input_sequences:
        :param output_sequences:
        :param embedding_matrix:
        :param max_sequence_len:
        :param actual_vocab_size:
        :return:
        '''
        print(len(input_sequences), len(output_sequences))
        print(embedding_matrix.shape, max_sequence_len, actual_vocab_size)
        print(input_sequences[0], output_sequences[0])
        print(input_sequences[1], output_sequences[1])

        one_hot_targets = np.zeros((len(input_sequences), max_sequence_len, actual_vocab_size))
        for i, target_sequence in enumerate(output_sequences):
            for t, word in enumerate(target_sequence):
                if word > 0:
                    one_hot_targets[i, t, word] = 1

        # create an embedding layer
        embedding_layer = Embedding(actual_vocab_size,
                                    config.EMBEDDING_DIM,
                                    weights=[embedding_matrix])

        # encoder
        input_layer = Input(shape=(max_sequence_len,))
        # we want to have control over the initial hidden states
        # as this is what will be based to the decoder
        initial_h = Input(shape=(config.HIDDEN_DIM,))
        initial_c = Input(shape=(config.HIDDEN_DIM,))
        x = embedding_layer(input_layer)
        lstm_layer = LSTM(config.HIDDEN_DIM, return_sequences=True, return_state=True, activation='tanh')
        x, _, _ = lstm_layer(x, initial_state=[initial_h, initial_c])
        dense_layer = Dense(actual_vocab_size, activation='softmax')
        output_layer = dense_layer(x)
        # to get probability distribution of the words
        # this model is being trained to generate the encoding
        # in such a way that it can predict the next word
        # this is why the output target is a one hot encoded and output layer is a dense layer

        encoder_model = Model([input_layer, initial_h, initial_c],
                              output_layer)

        encoder_model.compile(loss=config.LOSS_FN,
                              optimizer=Adam(lr=0.001),
                              metrics=['accuracy'])

        z = np.zeros((len(input_sequences), config.HIDDEN_DIM))
        r = encoder_model.fit(
            [input_sequences, z, z],
            one_hot_targets,
            batch_size=config.BATCH_SIZE,
            epochs=config.NUM_EPOCHS,
            validation_split=config.VALIDATION_SPLIT
        )
        print(r.history.keys())
        # plot some data
        plt.plot(r.history['loss'], label='loss')
        plt.plot(r.history['val_loss'], label='val_loss')
        plt.legend()
        plt.grid()
        plt.savefig('../plots/loss_values.png')
        plt.show()

        # accuracies
        plt.plot(r.history['acc'], label='acc')
        plt.plot(r.history['val_acc'], label='val_acc')
        plt.legend()
        plt.grid()
        plt.savefig('../plots/accuracy_values.png')
        plt.show()

        # create decoder
        # this will be a sampling model
        # given an input word, we will get a probability distribution of the output words
        # and we will randomly sample from this distribution to get the next word
        # the embedding and lstm layer that will be used is same as the encoder
        input_decoder = Input(shape=(1,))
        x = embedding_layer(input_decoder)
        x, h, c = lstm_layer(x, initial_state=[initial_h, initial_c])
        output_layer_decoder = dense_layer(x)
        sampling_model = Model([input_decoder, initial_h, initial_c],
                               [output_layer_decoder, h, c])


        return  encoder_model, sampling_model


    def sample_text_gen(self, sampling_model, word2idx, idx2word):
        '''

        :param sampling_model:
        :param idx2word:
        :return:
        '''
        # provide <sos> as first input
        np_sos_input = np.asarray([[word2idx['<SOS>']]])
        h = np.zeros((1, config.EMBEDDING_DIM))
        c = np.zeros((1, config.EMBEDDING_DIM))

        eos_id = word2idx['<EOS>']

        # for storing output
        output_sentence = []

        for _ in range(config.MAX_SEQUENCE_LENGTH):
            o, h, c = sampling_model.predict([np_sos_input, h, c])
            #TODO: continue with the prediction code. Lets train the model first


        return