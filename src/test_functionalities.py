""" Unit test cases for this project"""

# import modules here
import config
import os
import pickle
import feature_engg


class TestFunctionalities:

    # test if the file exists
    def test_raw_file_exists(self):
        assert True if os.path.isfile(config.RAW_DATASET) else False

    # test if the word embeddings exists
    def test_pretrained_embedding_exists(self):
        assert True if os.path.isfile(config.PRETRAINED_EMBEDDINGS)\
            else False

    # test if the input and output have start and end sequences
    def test_input_output_sequences(self):
        dl_obj = feature_engg.DumpLoadFile()
        tokenizer = dl_obj.load_file(config.TOKENIZER)
        assert True if tokenizer[0].word_index['<sos>'] and\
                       tokenizer[0].word_index['<eos>'] \
            else False

    # test if the word embedding matrix is of expected shape
    def test_embedding_matrix_shape(self):
        dl_obj = feature_engg.DumpLoadFile()
        embedding_matrix = dl_obj.load_file(config.EMBEDDING_MATRIX)
        assert True if embedding_matrix[0].shape[1] == config.EMBEDDING_DIM else False


