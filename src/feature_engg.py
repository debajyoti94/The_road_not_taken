''' Here we preprocess the text which will be used for training the model'''

# import modules here
import config

import abc
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pickle

class MustHaveDP:

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def tokenize_input(self):
        '''
        For tokenizing input and output data
        :return: tokenized data
        '''
        return

    @abc.abstractmethod
    def create_embedding_matrix(self):
        '''
        For creating the embedding matrix for fixed size input
        :return: embedding matrix
        '''
        return

class DataPreprocessing(MustHaveDP):

    # tokenize the input/output
    def tokenize_input(self, dataset):
        '''

        :param dataset: input dataset
        :return:    input sequences, output sequences, tokenizer
        '''
        # these lists will contain input/output texts
        # with <SOS> and <EOS> tags
        input_text = []
        output_text = []

        for line in open(dataset):
            line = line.strip()

            if not line:
                continue
            # adding the sos and eos tags
            input_line = '<SOS>' + line
            output_line = line + '<EOS'

            input_text.append(input_line)
            output_text.append(output_line)

        full_texts = input_text + output_text
        # we need full texts to capture the EOS and SOS tags and assign an index to them
        tokenizer = Tokenizer(num_words=config.MAX_VOCAB_SIZE)
        tokenizer.fit_on_texts(full_texts)
        # we have assigned an index value for each word
        # now we need to assign these indices to the sequences
        input_sequences = tokenizer.texts_to_sequences(input_text)
        output_sequences = tokenizer.texts_to_sequences(output_text)

        # now we need to create input and output sequences of fixed length
        max_sequence_len = max(len(s) for s in input_sequences)
        max_sequence_len = min(max_sequence_len,
                               config.MAX_SEQUENCE_LENGTH)
        input_sequences = pad_sequences(input_sequences,
                                        maxlen=max_sequence_len,
                                        padding='post')
        output_sequences = pad_sequences(output_sequences,
                                        maxlen=max_sequence_len,
                                        padding='post')
        # post indicates we will be
        # padding at the end of the sentence

        return input_sequences, output_sequences, tokenizer

    def load_glove_vectors(self, pretrained_model):
        '''
        In this function we load the glove vectors
        for the words that exist in the dataset.
        :param pretrained_model:
        :return: a dictionary with the word and pretrained vectors
        '''
        word_vector_dict = {}
        with open(config.PRETRAINED_EMBEDDINGS) as word_vectors_handle:
            for line in word_vectors_handle:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype=float)
                word_vector_dict[word] = vector

        return word_vector_dict


    def create_embedding_matrix(self, tokenizer, word_vector_dict):
        '''

        :param tokenizer:
        :param word_vector_dict:
        :return:
        '''

        # here we will create the word embedding matrix
        # basically, we will get the matrix of words and vectors
        # which exist in the voacbulary of the dataset

        word2idx = tokenizer.word_index
        num_words = min(config.MAX_VOCAB_SIZE, len(word2idx)+1) # +1 bcoz the index in tokenizer starts at 1
        embedding_matrix = np.zeros(shape=(num_words, config.EMBEDDING_DIM))

        for word, index in word2idx.items():
            try:
                embedding_matrix[index] = word_vector_dict.get(word)
            except KeyError:
                continue

        return embedding_matrix


class DumpLoadFile:

    def load_file(self, filename):
        '''

        :param filename:
        :return:
        '''
        with open(filename, 'rb') as pickle_handle:
            return pickle.load(pickle_handle)

    def dump_file(self, filename, *file):
        '''

        :param filename:
        :param file:
        :return:
        '''
        with open(filename, 'wb') as pickle_handle:
            pickle.dump(file)