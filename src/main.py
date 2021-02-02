''' Here we will call all the functions for training the model and
generating text by passing commandline arguments'''

import config
import feature_engg
import model

import argparse

from keras.models import load_model, save_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # args for data preprocessing
    parser.add_argument('--preprocess', type=str,
                        help='Provide arguments \"--preprocess data\" to extract features from the text.')

    # args for training the model
    parser.add_argument('--train', type=str,
                        help='Provide arguments \"--train seq2seq\" to train the seq2seq model.')

    # args for generating text
    parser.add_argument('--generate', type=str,
                        help='Provide arguments \"--generate text\" to generate text using the seq2seq model.')

    args = parser.parse_args()
    # creating obj for dumping and loading pickled files
    dl_obj = feature_engg.DumpLoadFile()

    if args.preprocess == 'data':
        # call the functions in feature_engg class
        dp_obj = feature_engg.DataPreprocessing()

        # get the input and output sequences
        print('Tokenizing input data ...')
        input_sequences, output_sequences, \
        tokenizer, max_sequence_len = dp_obj.tokenize_input(config.RAW_DATASET)

        # get the word vector dictionary
        print('Creating word vector dictionary ...')
        word_vector_dict = dp_obj.load_glove_vectors()

        # create the embedding matrix
        print('Creating the embedding matrix ...')
        embedding_matrix, num_words = dp_obj.create_embedding_matrix(tokenizer, word_vector_dict)

        print('Dumping the necessary files in pickle format ...')
        dl_obj.dump_file(config.INPUT_SEQ_TRAIN, input_sequences)
        dl_obj.dump_file(config.OUTPUT_SEQ_TRAIN, output_sequences)
        dl_obj.dump_file(config.TOKENIZER, tokenizer)
        dl_obj.dump_file(config.ACTUAL_SEQ_LENGTH, max_sequence_len)

        dl_obj.dump_file(config.WORD_VECTOR_DICT, word_vector_dict)

        dl_obj.dump_file(config.EMBEDDING_MATRIX, embedding_matrix)
        dl_obj.dump_file(config.ACTUAL_VOCAB_SIZE, num_words)

    elif args.train == 'seq2seq':

        # we will load all the necessary files required for training the model
        input_sequences = dl_obj.load_file(config.INPUT_SEQ_TRAIN)
        output_sequences = dl_obj.load_file(config.OUTPUT_SEQ_TRAIN)
        embedding_matrix = dl_obj.load_file(config.EMBEDDING_MATRIX)
        actual_sequence_len = dl_obj.load_file(config.ACTUAL_SEQ_LENGTH)
        actual_vocab_size = dl_obj.load_file(config.ACTUAL_VOCAB_SIZE)

        lm = model.LanguageModel()
        print(actual_vocab_size, actual_sequence_len)
        encoder_model, sampling_model = lm.train_model(input_sequences[0],
                                                       output_sequences[0],
                                                       embedding_matrix[0],
                                                       actual_sequence_len[0],
                                                       actual_vocab_size[0])

        save_model(encoder_model, config.ENCODER_MODEL)
        save_model(sampling_model, config.SAMPLING_MODEL)

    elif args.generate == 'text':

        sampling_model = load_model(config.SAMPLING_MODEL)
        tokenizer = dl_obj.load_file(config.TOKENIZER)


        lm = model.LanguageModel()
        while True:
            for _ in range(5):
                print(lm.sample_text_gen(sampling_model, tokenizer[0].word_index,
                           tokenizer[0].index_word))

            ans = input("---generate another? [Y/n]---")
            if ans and ans[0].lower().startswith('n'):
                break


