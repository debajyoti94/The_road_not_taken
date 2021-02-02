""" Unit test cases for this project"""

# import modules here
import config
import os
import pickle


class TestFunctionalities:

    # test if the file exists
    def test_raw_file_exists(self):
        assert True if os.path.isfile(config.RAW_DATASET) else False


# test if the input and out have start and end sequences


# test if the word embedding matrix is of expected shape


# test if the word embeddings exists
